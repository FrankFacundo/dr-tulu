from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, get_scheduler

from open_instruct2.bootstrap import ensure_open_instruct_importable
from open_instruct2.config import TrainArgs
from open_instruct2.reference_vllm import (
    VLLMReferenceClient,
    VLLMServerProcess,
    create_reference_client,
)
from open_instruct2.rewards import RewardEngine
from open_instruct2.runtime import RuntimeInfo, resolve_runtime, set_seed

ensure_open_instruct_importable()

from open_instruct.dataset_transformation import (  # noqa: E402
    DATASET_ORIGIN_KEY,
    DATASET_SOURCE_KEY,
    GROUND_TRUTHS_KEY,
    INPUT_IDS_PROMPT_KEY,
    RAW_USER_QUERY,
    TokenizerConfig,
    get_cached_dataset_tulu,
)
from open_instruct.ground_truth_utils import cleanup_all_llm_judge_clients  # noqa: E402
from open_instruct.model_utils import disable_dropout_in_model  # noqa: E402

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from open_instruct2.memory_profile import MemoryProfiler


@dataclass
class RolloutBatch:
    query_responses: List[List[int]]
    responses: List[List[int]]
    response_masks: List[List[int]]
    finish_reasons: List[str]
    decoded_responses: List[str]
    ground_truths: List[Union[str, List[str]]]
    datasets: List[Union[str, List[str]]]
    queries: List[str]


class ShufflingIterator:
    def __init__(self, data: np.ndarray, batch_size: int, seed: Optional[int] = None):
        self.data = data.copy()
        self.batch_size = batch_size
        self.index = 0
        self.rng = np.random.default_rng(seed)
        self.rng.shuffle(self.data)
        self.effective_size = len(self.data) - (len(self.data) % batch_size)
        if self.effective_size == 0:
            raise ValueError("Dataset is too small for num_unique_prompts_rollout.")

    def __iter__(self) -> Iterator[List[int]]:
        return self

    def __next__(self) -> List[int]:
        if self.index >= self.effective_size:
            self.index = 0
            self.rng.shuffle(self.data)
        end_index = self.index + self.batch_size
        batch = self.data[self.index : end_index].tolist()
        self.index = end_index
        return batch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (values * mask_f).sum() / denom


def _compute_advantages(
    scores: np.ndarray,
    num_samples_per_prompt: int,
    normalization_type: str,
) -> np.ndarray:
    scores_per_prompt = scores.reshape(-1, num_samples_per_prompt)
    mean_grouped = scores_per_prompt.mean(axis=-1, keepdims=True)
    std_grouped = scores_per_prompt.std(axis=-1, keepdims=True)
    if normalization_type == "standard":
        advantages = (scores_per_prompt - mean_grouped) / (std_grouped + 1e-8)
    elif normalization_type == "centered":
        advantages = scores_per_prompt - mean_grouped
    else:
        raise ValueError(f"Unknown advantage normalization type: {normalization_type}")
    return advantages.reshape(-1)


def _infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    candidate_suffixes = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "query_key_value",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    found: List[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = module_name.split(".")[-1]
        if suffix in candidate_suffixes and suffix not in found:
            found.append(suffix)
    if found:
        return found
    # Fallback: last linear names with broad compatibility.
    fallback = []
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            suffix = module_name.split(".")[-1]
            if suffix not in fallback:
                fallback.append(suffix)
    if not fallback:
        raise RuntimeError("Could not infer LoRA target modules from policy model.")
    return fallback[:8]


class SingleProcessGRPOTrainer:
    def __init__(self, args: TrainArgs) -> None:
        self.args = args
        if self.args.run_name is None:
            if self.args.smoke_test:
                self.args.run_name = f"{self.args.exp_name}__smoke__{self.args.seed}__{int(time.time())}"
            else:
                self.args.run_name = f"{self.args.exp_name}__{self.args.seed}__{int(time.time())}"
        self.output_dir = Path(self.args.output_dir) / self.args.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

        self._setup_logging()
        self.memory_profiler: Optional["MemoryProfiler"] = None
        if self.args.enable_memory_profile:
            try:
                from open_instruct2.memory_profile import MemoryProfiler
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "Memory profiling requires `psutil`. Install dependencies with `pip install -e rl/open-instruct2`."
                ) from exc
            self.memory_profiler = MemoryProfiler(
                output_dir=self.output_dir / "memory_profile",
                interval_s=self.args.memory_profile_interval_s,
                top_n_children=self.args.memory_profile_top_n_children,
                enable_tracemalloc=self.args.memory_profile_enable_tracemalloc,
                logger=LOGGER,
            )
            self.memory_profiler.start()
            self.memory_profiler.mark("trainer_init_start")

        self.runtime: RuntimeInfo = resolve_runtime(self.args)
        set_seed(self.args.seed)
        LOGGER.info(
            "Runtime resolved: device=%s dtype=%s attn=%s",
            self.runtime.device,
            self.runtime.dtype_str,
            self.runtime.attn_implementation,
        )
        self._mark_memory("runtime_resolved")
        if self.args.smoke_test:
            LOGGER.info(
                "Smoke test mode enabled: steps=%d rollout=(%d prompts x %d samples) response_length=%d",
                self.args.smoke_test_steps,
                self.args.num_unique_prompts_rollout,
                self.args.num_samples_per_prompt_rollout,
                self.args.response_length,
            )

        self.tokenizer = self._build_tokenizer()
        self._mark_memory("tokenizer_ready")
        self.train_dataset, self.eval_dataset = self._load_datasets()
        self._mark_memory("datasets_ready")
        if self.args.cache_dataset_only:
            LOGGER.info("Dataset cache-only mode enabled; exiting after cache fill.")
            self.writer = None
            self.policy = None
            self.optimizer = None
            self.scheduler = None
            self.reference_client = None
            self.reference_process = None
            self.reward_engine = None
            self.rubric_buffer = None
            self.eval_freq = -1
            self.wandb_run = None
            self._mark_memory("cache_dataset_only_exit")
            return

        self.num_training_steps = self._resolve_num_training_steps()
        self.eval_freq = self._resolve_eval_freq()

        self.policy = self._load_policy_model()
        self._mark_memory("policy_ready")
        self.optimizer, self.scheduler = self._configure_optimizer()
        self.reference_client, self.reference_process = create_reference_client(
            self.args,
            self.runtime,
            self.tokenizer,
            str(self.output_dir),
        )
        self._mark_memory("reference_ready")
        self.reward_engine = RewardEngine(self.args)
        self.rubric_buffer = self.reward_engine.initialize_rubric_buffer(self.train_dataset[GROUND_TRUTHS_KEY])
        self._mark_memory("reward_engine_ready")

        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self.writer.add_text(
            "run_config",
            json.dumps({k: str(v) for k, v in vars(self.args).items()}, indent=2),
        )

        self.wandb_run = None
        if self.args.with_tracking:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=self.args.wandb_project_name,
                    entity=self.args.wandb_entity,
                    name=self.args.run_name,
                    config=vars(self.args),
                )
            except Exception as exc:
                LOGGER.warning("Failed to initialize wandb tracking: %s", exc)

        self.train_iter = ShufflingIterator(
            np.arange(len(self.train_dataset)),
            batch_size=self.args.num_unique_prompts_rollout,
            seed=self.args.seed,
        )
        self.num_total_tokens = 0
        self.start_time = time.time()
        self._mark_memory("trainer_init_done")

    def _setup_logging(self) -> None:
        logging.basicConfig(
            level=getattr(logging, self.args.log_level.upper(), logging.INFO),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    def _mark_memory(self, label: str) -> None:
        if self.memory_profiler is not None:
            self.memory_profiler.mark(label)

    def _maybe_mark_step_memory(self, training_step: int, suffix: str) -> None:
        if self.memory_profiler is None or not self.args.memory_profile_mark_steps:
            return
        if training_step % self.args.memory_profile_step_interval != 0:
            return
        self.memory_profiler.mark(f"step_{training_step}_{suffix}")

    def _build_tokenizer(self):
        tokenizer_name = self.args.tokenizer_name_or_path or self.args.model_name_or_path
        tokenizer_revision = self.args.tokenizer_revision or self.args.model_revision
        tc = TokenizerConfig(
            tokenizer_name_or_path=tokenizer_name,
            tokenizer_revision=tokenizer_revision,
            trust_remote_code=self.args.trust_remote_code,
            use_fast=self.args.use_fast_tokenizer,
            chat_template_name=self.args.chat_template_name,
            add_bos=self.args.add_bos,
            get_tokenizer_fn=self.args.tokenizer_fn,
        )
        tokenizer = tc.tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define eos_token_id to derive pad_token_id.")
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_system_prompts(self) -> tuple[Optional[str], Optional[str]]:
        system_prompt_text = None
        additional_question_instructions = None
        if self.args.system_prompt_file is None:
            return system_prompt_text, additional_question_instructions

        path = Path(self.args.system_prompt_file)
        if not path.exists():
            raise ValueError(f"System prompt file not found: {path}")
        if path.suffix == ".txt":
            system_prompt_text = path.read_text(encoding="utf-8").strip()
            return system_prompt_text, additional_question_instructions
        if path.suffix == ".yaml":
            import yaml

            prompt = yaml.safe_load(path.read_text(encoding="utf-8"))
            system_prompt_text = prompt["system_prompt"]
            additional_question_instructions = prompt.get("additional_instructions")
            return system_prompt_text, additional_question_instructions
        raise ValueError(f"Unsupported system prompt file format: {path.suffix}")

    def _load_datasets(self):
        system_prompt_text, additional_question_instructions = self._load_system_prompts()
        transform_fn_args = [
            {
                "system_prompt_text": system_prompt_text,
                "additional_question_instructions": additional_question_instructions,
            },
            {
                "max_token_length": self.args.max_token_length,
                "max_prompt_token_length": self.args.max_prompt_token_length,
            },
        ]
        train_dataset = get_cached_dataset_tulu(
            dataset_mixer_list=self.args.dataset_mixer_list,
            dataset_mixer_list_splits=self.args.dataset_mixer_list_splits,
            tc=TokenizerConfig(
                tokenizer_name_or_path=self.args.tokenizer_name_or_path or self.args.model_name_or_path,
                tokenizer_revision=self.args.tokenizer_revision or self.args.model_revision,
                trust_remote_code=self.args.trust_remote_code,
                use_fast=self.args.use_fast_tokenizer,
                chat_template_name=self.args.chat_template_name,
                add_bos=self.args.add_bos,
                get_tokenizer_fn=self.args.tokenizer_fn,
            ),
            dataset_transform_fn=self.args.dataset_transform_fn,
            transform_fn_args=transform_fn_args,
            dataset_cache_mode=self.args.dataset_cache_mode,
            dataset_config_hash=self.args.dataset_config_hash,
            dataset_local_cache_dir=os.path.abspath(self.args.dataset_local_cache_dir),
            dataset_skip_cache=self.args.dataset_skip_cache,
        ).shuffle(seed=self.args.seed)
        eval_dataset = None
        if len(self.args.dataset_mixer_eval_list) > 0:
            eval_dataset = get_cached_dataset_tulu(
                dataset_mixer_list=self.args.dataset_mixer_eval_list,
                dataset_mixer_list_splits=self.args.dataset_mixer_eval_list_splits,
                tc=TokenizerConfig(
                    tokenizer_name_or_path=self.args.tokenizer_name_or_path or self.args.model_name_or_path,
                    tokenizer_revision=self.args.tokenizer_revision or self.args.model_revision,
                    trust_remote_code=self.args.trust_remote_code,
                    use_fast=self.args.use_fast_tokenizer,
                    chat_template_name=self.args.chat_template_name,
                    add_bos=self.args.add_bos,
                    get_tokenizer_fn=self.args.tokenizer_fn,
                ),
                dataset_transform_fn=self.args.dataset_transform_fn,
                transform_fn_args=transform_fn_args,
                dataset_cache_mode=self.args.dataset_cache_mode,
                dataset_config_hash=self.args.dataset_config_eval_hash,
                dataset_local_cache_dir=os.path.abspath(self.args.dataset_local_cache_dir),
                dataset_skip_cache=self.args.dataset_skip_cache,
            )
            if self.args.shuffle_eval_dataset:
                eval_dataset = eval_dataset.shuffle(seed=self.args.seed)
        LOGGER.info("Loaded datasets: train=%d eval=%s", len(train_dataset), len(eval_dataset) if eval_dataset else 0)
        return train_dataset, eval_dataset

    def _resolve_num_training_steps(self) -> int:
        if self.args.num_training_steps is not None and self.args.num_training_steps > 0:
            return self.args.num_training_steps
        denominator = self.args.num_unique_prompts_rollout * self.args.num_samples_per_prompt_rollout
        steps = self.args.total_episodes // denominator
        if steps <= 0:
            raise ValueError("num_training_steps resolved to <= 0; increase total_episodes.")
        return steps

    def _resolve_eval_freq(self) -> int:
        if self.args.num_evals <= 0:
            return -1
        return max(1, self.num_training_steps // self.args.num_evals)

    def _load_policy_model(self):
        load_kwargs = dict(
            revision=self.args.model_revision,
            trust_remote_code=self.args.trust_remote_code,
            torch_dtype=self.runtime.torch_dtype,
            use_cache=False,
        )
        if self.args.attn_implementation or self.runtime.device_type == "cuda":
            load_kwargs["attn_implementation"] = self.runtime.attn_implementation
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                **load_kwargs,
            )
        except TypeError:
            load_kwargs.pop("attn_implementation", None)
            model = AutoModelForCausalLM.from_pretrained(
                self.args.model_name_or_path,
                **load_kwargs,
            )
        model.to(self.runtime.device)
        disable_dropout_in_model(model)
        if self.args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if self.args.train_mode == "lora":
            from peft import LoraConfig, TaskType, get_peft_model

            target_modules = self.args.lora_target_modules or _infer_lora_target_modules(model)
            lora_cfg = LoraConfig(
                r=self.args.lora_r,
                lora_alpha=self.args.lora_alpha,
                lora_dropout=self.args.lora_dropout,
                target_modules=target_modules,
                task_type=TaskType.CAUSAL_LM,
            )
            model = get_peft_model(model, lora_cfg)
            model.print_trainable_parameters()
            LOGGER.info("LoRA enabled with target modules: %s", target_modules)
        return model

    def _configure_optimizer(self):
        trainable_params = [p for p in self.policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        samples_per_step = self.args.num_unique_prompts_rollout * self.args.num_samples_per_prompt_rollout
        mini_batch_size = max(
            1,
            min(
                self.args.per_device_train_batch_size,
                math.ceil(samples_per_step / self.args.num_mini_batches),
            ),
        )
        optimization_steps_per_rollout = math.ceil(samples_per_step / mini_batch_size) * self.args.num_epochs
        num_scheduler_steps = self.num_training_steps * optimization_steps_per_rollout
        warmup_steps = self.args.warm_up_steps
        if self.args.warmup_ratio > 0.0:
            warmup_steps = int(num_scheduler_steps * self.args.warmup_ratio)
        scheduler = get_scheduler(
            self.args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_scheduler_steps,
        )
        return optimizer, scheduler

    def _forward_logprobs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        inputs = torch.where(attention_mask.bool(), input_ids, torch.zeros_like(input_ids))
        outputs = self.policy(
            input_ids=inputs[:, :-1],
            attention_mask=attention_mask[:, :-1],
            return_dict=True,
        )
        logits = outputs.logits / (self.args.temperature + 1e-7)
        target_ids = inputs[:, 1:].unsqueeze(-1)
        logprobs = F.log_softmax(logits, dim=-1).gather(-1, target_ids).squeeze(-1)
        return logprobs

    def _compute_policy_logprobs_list(self, sequences: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        self.policy.eval()
        out: List[torch.Tensor] = []
        batch_size = max(1, self.args.per_device_train_batch_size)
        with torch.no_grad():
            for start in range(0, len(sequences), batch_size):
                chunk = sequences[start : start + batch_size]
                chunk_tensors = [torch.tensor(seq, dtype=torch.long, device=self.runtime.device) for seq in chunk]
                padded = pad_sequence(chunk_tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id)
                attention_mask = padded.ne(self.tokenizer.pad_token_id).long()
                logprobs = self._forward_logprobs(padded, attention_mask).detach().cpu()
                lengths = [len(seq) - 1 for seq in chunk]
                for i, length in enumerate(lengths):
                    out.append(logprobs[i, :length])
        self.policy.train()
        return out

    def _compute_reference_logprobs_list(self, sequences: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        assert isinstance(self.reference_client, VLLMReferenceClient)
        return self.reference_client.score_sequences(sequences)

    def _pad_batch(
        self,
        tensor_lists: Sequence[Sequence[int]],
        pad_value: int,
        device: torch.device,
    ) -> torch.Tensor:
        tensors = [torch.tensor(seq, dtype=torch.long) for seq in tensor_lists]
        padded = pad_sequence(tensors, batch_first=True, padding_value=pad_value)
        return padded.to(device)

    def _pad_float_batch(self, tensors: Sequence[torch.Tensor], device: torch.device) -> torch.Tensor:
        padded = pad_sequence([t.float() for t in tensors], batch_first=True, padding_value=0.0)
        return padded.to(device)

    def _generate_rollouts(
        self,
        prompts: List[List[int]],
        ground_truths: List[Union[str, List[str]]],
        datasets: List[Union[str, List[str]]],
        raw_queries: List[str],
        num_samples_per_prompt: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> RolloutBatch:
        self.policy.eval()
        num_samples = num_samples_per_prompt or self.args.num_samples_per_prompt_rollout
        temperature = self.args.temperature if temperature is None else temperature
        prompt_tensors = [torch.tensor(p, dtype=torch.long) for p in prompts]
        input_ids = pad_sequence(prompt_tensors, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(
            self.runtime.device
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            generated = self.policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=temperature,
                top_p=self.args.top_p,
                max_new_tokens=self.args.response_length,
                num_return_sequences=num_samples,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        prompt_lengths = attention_mask.sum(dim=1).tolist()
        expanded_prompt_lengths = [length for length in prompt_lengths for _ in range(num_samples)]
        expanded_ground_truths = [item for item in ground_truths for _ in range(num_samples)]
        expanded_datasets = [item for item in datasets for _ in range(num_samples)]
        expanded_queries = [item for item in raw_queries for _ in range(num_samples)]

        query_responses: List[List[int]] = []
        responses: List[List[int]] = []
        response_masks: List[List[int]] = []
        finish_reasons: List[str] = []

        for i, sequence in enumerate(generated.tolist()):
            prompt_len = expanded_prompt_lengths[i]
            query_tokens = sequence[:prompt_len]
            response_tokens = sequence[prompt_len:]
            if len(response_tokens) == 0 and self.tokenizer.eos_token_id is not None:
                response_tokens = [self.tokenizer.eos_token_id]
            reason = "stop" if (self.tokenizer.eos_token_id in response_tokens) else "length"
            if self.args.stop_strings:
                decoded_resp = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                if any(stop_str in decoded_resp for stop_str in self.args.stop_strings):
                    reason = "stop"
            query_response = query_tokens + response_tokens
            response_mask = [0] * len(query_tokens) + [1] * len(response_tokens)

            query_responses.append(query_response)
            responses.append(response_tokens)
            response_masks.append(response_mask)
            finish_reasons.append(reason)

        decoded_responses = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        self.policy.train()

        return RolloutBatch(
            query_responses=query_responses,
            responses=responses,
            response_masks=response_masks,
            finish_reasons=finish_reasons,
            decoded_responses=decoded_responses,
            ground_truths=expanded_ground_truths,
            datasets=expanded_datasets,
            queries=expanded_queries,
        )

    def _train_on_rollout(
        self,
        rollout: RolloutBatch,
        advantages: np.ndarray,
    ) -> Dict[str, float]:
        if len(rollout.query_responses) == 0:
            return {}

        old_logprobs = self._compute_policy_logprobs_list(rollout.query_responses)
        ref_logprobs = self._compute_reference_logprobs_list(rollout.query_responses)

        batch_size = max(
            1,
            min(
                self.args.per_device_train_batch_size,
                math.ceil(len(rollout.query_responses) / self.args.num_mini_batches),
            ),
        )
        indices = np.arange(len(rollout.query_responses))

        metrics = defaultdict(list)
        self.policy.train()
        global_mb_step = 0
        self.optimizer.zero_grad(set_to_none=True)

        for epoch_idx in range(self.args.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                mb_indices = indices[start : start + batch_size]
                if len(mb_indices) == 0:
                    continue
                mb_query_responses = [rollout.query_responses[i] for i in mb_indices]
                mb_response_masks = [rollout.response_masks[i] for i in mb_indices]
                mb_old_logprobs = [old_logprobs[i] for i in mb_indices]
                mb_ref_logprobs = [ref_logprobs[i] for i in mb_indices]
                mb_advantages = torch.tensor(
                    advantages[mb_indices],
                    dtype=torch.float32,
                    device=self.runtime.device,
                )

                query_responses = self._pad_batch(
                    mb_query_responses,
                    pad_value=self.tokenizer.pad_token_id,
                    device=self.runtime.device,
                )
                response_masks = self._pad_batch(
                    mb_response_masks,
                    pad_value=0,
                    device=self.runtime.device,
                )
                old_lp = self._pad_float_batch(mb_old_logprobs, device=self.runtime.device)
                ref_lp = self._pad_float_batch(mb_ref_logprobs, device=self.runtime.device)

                attention_mask = query_responses.ne(self.tokenizer.pad_token_id).long()
                new_lp = self._forward_logprobs(query_responses, attention_mask)
                response_mask_bool = response_masks[:, 1:].bool()

                target_len = new_lp.shape[1]
                if old_lp.shape[1] < target_len:
                    old_lp = F.pad(old_lp, (0, target_len - old_lp.shape[1]), value=0.0)
                old_lp = old_lp[:, :target_len]
                if ref_lp.shape[1] < target_len:
                    ref_lp = F.pad(ref_lp, (0, target_len - ref_lp.shape[1]), value=0.0)
                ref_lp = ref_lp[:, :target_len]
                if response_mask_bool.shape[1] < target_len:
                    response_mask_bool = F.pad(response_mask_bool.float(), (0, target_len - response_mask_bool.shape[1]), value=0.0).bool()
                response_mask_bool = response_mask_bool[:, :target_len]

                logprobs_diff = new_lp - old_lp
                ratio = torch.exp(logprobs_diff)
                adv = mb_advantages.unsqueeze(1)
                pg_losses = -adv * ratio
                pg_losses2 = -adv * torch.clamp(ratio, 1.0 - self.args.clip_lower, 1.0 + self.args.clip_higher)
                pg_loss_max = torch.max(pg_losses, pg_losses2)

                ref_logprobs_diff = (new_lp - ref_lp).clamp(-40.0, 40.0)
                kl1 = ref_logprobs_diff
                kl2 = (ref_logprobs_diff**2) / 2.0
                kl3 = torch.expm1(-ref_logprobs_diff) + ref_logprobs_diff
                kl4 = ratio * ref_logprobs_diff
                if self.args.kl_estimator == "kl1":
                    kl = kl1
                    kl_stat = kl1
                elif self.args.kl_estimator == "kl2":
                    kl = kl2
                    kl_stat = kl2
                elif self.args.kl_estimator == "kl3":
                    kl = kl3
                    kl_stat = kl3
                elif self.args.kl_estimator == "kl4":
                    kl = kl4
                    kl_stat = kl4
                else:
                    raise ValueError(f"Unsupported kl_estimator: {self.args.kl_estimator}")

                loss = masked_mean(pg_loss_max + (self.args.beta * kl), response_mask_bool)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy.parameters() if p.requires_grad],
                    self.args.max_grad_norm,
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                global_mb_step += 1

                with torch.no_grad():
                    metrics["objective/kl_avg"].append(masked_mean(kl1, response_mask_bool).item())
                    metrics["objective/kl2_avg"].append(masked_mean(kl2, response_mask_bool).item())
                    metrics["objective/kl3_avg"].append(masked_mean(kl3, response_mask_bool).item())
                    metrics["objective/kl4_avg"].append(masked_mean(kl4, response_mask_bool).item())
                    metrics["loss/policy_avg"].append(masked_mean(pg_loss_max, response_mask_bool).item())
                    metrics["loss/kl_avg"].append((self.args.beta * masked_mean(kl_stat, response_mask_bool)).item())
                    metrics["loss/total_avg"].append(loss.item())
                    metrics["policy/clipfrac_avg"].append(
                        masked_mean((pg_losses2 > pg_losses).float(), response_mask_bool).item()
                    )
                    metrics["val/ratio"].append(masked_mean(ratio, response_mask_bool).item())
                    metrics["val/ratio_var"].append(
                        ((ratio * response_mask_bool.float()).var(unbiased=False)).item()
                    )
                    metrics["lr"].append(float(self.scheduler.get_last_lr()[0]))

        scalar_metrics = {key: float(np.mean(values)) for key, values in metrics.items() if len(values) > 0}
        scalar_metrics["train/num_minibatch_steps"] = float(global_mb_step)
        return scalar_metrics

    def _log_step_metrics(self, metrics: Dict[str, float], episode: int) -> None:
        if self.memory_profiler is not None:
            metrics = dict(metrics)
            metrics.update(self.memory_profiler.latest_metrics_gb())
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.floating)):
                self.writer.add_scalar(key, float(value), episode)
        if self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=episode)
        summary = " | ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float)))
        LOGGER.info(summary)

    def _save_adaptive_rubrics(self, training_step: int, payload: List[Dict]) -> None:
        path = self.output_dir / f"adaptive_rubrics_{self.args.run_name}.jsonl"
        data = {
            "training_step": training_step,
            "adaptive_rubric_scores": payload,
        }
        with path.open("a", encoding="utf-8") as handle:
            json.dump(data, handle)
            handle.write("\n")

    def _save_model(self, output_path: Path) -> None:
        output_path.mkdir(parents=True, exist_ok=True)
        self.policy.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))

    def _cleanup_old_checkpoints(self) -> None:
        if self.args.keep_last_n_checkpoints < 0:
            return
        checkpoints = sorted(
            [path for path in self.checkpoint_dir.glob("step_*") if path.is_dir()],
            key=lambda path: int(path.name.split("_")[-1]),
        )
        while len(checkpoints) > self.args.keep_last_n_checkpoints:
            old = checkpoints.pop(0)
            shutil.rmtree(old, ignore_errors=True)

    def _evaluate(self, training_step: int, episode: int) -> Dict[str, float]:
        if self.eval_dataset is None:
            return {}
        n_eval = min(self.args.eval_max_samples, len(self.eval_dataset))
        eval_slice = self.eval_dataset.select(range(n_eval))

        prompts = eval_slice[INPUT_IDS_PROMPT_KEY]
        ground_truths = eval_slice[GROUND_TRUTHS_KEY]
        datasets = eval_slice[DATASET_SOURCE_KEY]
        queries = eval_slice[RAW_USER_QUERY] if RAW_USER_QUERY in eval_slice.column_names else [""] * n_eval

        rollout = self._generate_rollouts(
            prompts=prompts,
            ground_truths=ground_truths,
            datasets=datasets,
            raw_queries=queries,
            num_samples_per_prompt=1,
            temperature=0.6,
        )

        source_datasets = (
            eval_slice[DATASET_ORIGIN_KEY]
            if DATASET_ORIGIN_KEY in eval_slice.column_names
            else None
        )
        eval_scores, eval_reward_metrics, _, _ = asyncio.run(
            self.reward_engine.compute(
                rollout.responses,
                rollout.decoded_responses,
                rollout.ground_truths,
                rollout.datasets,
                rollout.finish_reasons,
                queries=rollout.queries,
                source_datasets=source_datasets,
                rubric_buffer=None,
                is_training=False,
                training_step=training_step,
            )
        )

        eval_sequence_lengths = np.array([len(r) for r in rollout.responses], dtype=np.int32)
        eval_stop_rate = float(np.mean([reason == "stop" for reason in rollout.finish_reasons]))
        eval_metrics = {
            "eval/scores": float(np.mean(eval_scores)) if len(eval_scores) > 0 else 0.0,
            "eval/sequence_lengths": float(eval_sequence_lengths.mean()) if len(eval_sequence_lengths) > 0 else 0.0,
            "eval/sequence_lengths_min": float(eval_sequence_lengths.min()) if len(eval_sequence_lengths) > 0 else 0.0,
            "eval/sequence_lengths_max": float(eval_sequence_lengths.max()) if len(eval_sequence_lengths) > 0 else 0.0,
            "eval/stop_rate": eval_stop_rate,
        }
        eval_metrics.update({f"eval/{k}": float(v) for k, v in eval_reward_metrics.items()})

        per_dataset = defaultdict(list)
        for ds_name, score in zip(rollout.datasets, eval_scores):
            per_dataset[str(ds_name)].append(float(score))
        if per_dataset:
            means = []
            for ds_name, values in per_dataset.items():
                mean_val = float(np.mean(values))
                eval_metrics[f"eval/scores_{ds_name}"] = mean_val
                means.append(mean_val)
            eval_metrics["eval/scores_macro"] = float(np.mean(means))

        self._log_step_metrics(eval_metrics, episode=episode)
        return eval_metrics

    def run(self) -> None:
        if self.args.cache_dataset_only:
            return

        LOGGER.info("Starting training for %d steps.", self.num_training_steps)
        self._mark_memory("training_start")
        episode = 0
        for training_step in range(1, self.num_training_steps + 1):
            self._maybe_mark_step_memory(training_step, "start")
            batch_indices = next(self.train_iter)
            batch_data = self.train_dataset[batch_indices]
            prompts = batch_data[INPUT_IDS_PROMPT_KEY]
            ground_truths = batch_data[GROUND_TRUTHS_KEY]
            datasets = batch_data[DATASET_SOURCE_KEY]
            raw_queries = (
                batch_data[RAW_USER_QUERY]
                if RAW_USER_QUERY in batch_data
                else ["" for _ in range(len(prompts))]
            )

            added_static = self.reward_engine.maybe_add_static_rubrics(self.rubric_buffer, training_step)
            if added_static:
                LOGGER.info("Added %d static rubrics back into active rubric buffer.", added_static)

            rollout = self._generate_rollouts(prompts, ground_truths, datasets, raw_queries)
            if len(rollout.responses) == 0:
                LOGGER.warning("Skipping step %d due to empty rollout responses.", training_step)
                continue
            self._maybe_mark_step_memory(training_step, "after_rollout")

            scores, reward_metrics, self.rubric_buffer, adaptive_rubrics = asyncio.run(
                self.reward_engine.compute(
                    rollout.responses,
                    rollout.decoded_responses,
                    rollout.ground_truths,
                    rollout.datasets,
                    rollout.finish_reasons,
                    queries=rollout.queries,
                    rubric_buffer=self.rubric_buffer,
                    is_training=True,
                    training_step=training_step,
                )
            )
            self._maybe_mark_step_memory(training_step, "after_rewards")

            if self.args.save_adaptive_rubrics and adaptive_rubrics is not None:
                self._save_adaptive_rubrics(training_step, adaptive_rubrics)

            scores_np = np.array(scores, dtype=np.float32)
            scores_per_prompt = scores_np.reshape(-1, self.args.num_samples_per_prompt_rollout)
            std_per_prompt = scores_per_prompt.std(axis=-1)
            non_zero_std_mask = std_per_prompt != 0
            expanded_mask = np.repeat(non_zero_std_mask, self.args.num_samples_per_prompt_rollout)

            if self.args.mask_truncated_completions:
                stop_mask = np.array([reason == "stop" for reason in rollout.finish_reasons], dtype=bool)
                expanded_mask = expanded_mask & stop_mask

            selected_idx = np.where(expanded_mask)[0]
            if len(selected_idx) == 0:
                if self.args.smoke_test:
                    LOGGER.warning(
                        "Smoke test fallback at step %d: all groups had zero reward variance; using unfiltered rollout.",
                        training_step,
                    )
                    selected_idx = np.arange(len(scores_np))
                else:
                    LOGGER.warning("Skipping step %d because all groups had zero variance rewards.", training_step)
                    continue

            filtered_scores = scores_np[selected_idx]
            filtered_rollout = RolloutBatch(
                query_responses=[rollout.query_responses[i] for i in selected_idx],
                responses=[rollout.responses[i] for i in selected_idx],
                response_masks=[rollout.response_masks[i] for i in selected_idx],
                finish_reasons=[rollout.finish_reasons[i] for i in selected_idx],
                decoded_responses=[rollout.decoded_responses[i] for i in selected_idx],
                ground_truths=[rollout.ground_truths[i] for i in selected_idx],
                datasets=[rollout.datasets[i] for i in selected_idx],
                queries=[rollout.queries[i] for i in selected_idx],
            )

            # Advantages are computed before filtering, then filtered by same mask.
            advantages_full = _compute_advantages(
                scores_np,
                num_samples_per_prompt=self.args.num_samples_per_prompt_rollout,
                normalization_type=self.args.advantage_normalization_type,
            )
            filtered_advantages = advantages_full[selected_idx]

            train_metrics = self._train_on_rollout(filtered_rollout, filtered_advantages)
            self._maybe_mark_step_memory(training_step, "after_train")

            num_new_tokens = int(sum(len(response) for response in filtered_rollout.responses))
            self.num_total_tokens += num_new_tokens
            episode += self.args.num_unique_prompts_rollout * self.args.num_samples_per_prompt_rollout

            max_possible_score = 0.0
            if self.args.apply_verifiable_reward:
                max_possible_score += self.args.verification_reward
            if self.args.apply_r1_style_format_reward and self.args.additive_format_reward:
                max_possible_score += self.args.r1_style_format_reward

            sequence_lengths = np.array([len(response) for response in filtered_rollout.responses], dtype=np.int32)
            sequence_length_solved = (
                np.array([], dtype=np.int32)
                if np.all(filtered_scores == 0)
                else sequence_lengths[filtered_scores == max_possible_score]
            )
            sequence_length_unsolved = (
                np.array([], dtype=np.int32)
                if np.all(filtered_scores == max_possible_score)
                else sequence_lengths[filtered_scores == 0]
            )
            stop_rate = float(np.mean([r == "stop" for r in filtered_rollout.finish_reasons]))

            step_metrics = {
                "episode": float(episode),
                "training_step": float(training_step),
                "epoch": float(episode / self.args.num_samples_per_prompt_rollout / len(self.train_dataset)),
                "tokens_per_second": float(self.num_total_tokens / max(1e-6, time.time() - self.start_time)),
                "val/num_total_tokens": float(self.num_total_tokens),
                "scores": float(filtered_scores.mean()),
                "real_batch_size_ratio": float(len(selected_idx) / len(scores_np)),
                "packed_ratio": 1.0,
                "val/sequence_lengths": float(sequence_lengths.mean()) if len(sequence_lengths) > 0 else 0.0,
                "val/sequence_lengths_min": float(sequence_lengths.min()) if len(sequence_lengths) > 0 else 0.0,
                "val/sequence_lengths_max": float(sequence_lengths.max()) if len(sequence_lengths) > 0 else 0.0,
                "val/sequence_lengths_unsolved": float(sequence_length_unsolved.mean()) if len(sequence_length_unsolved) > 0 else 0.0,
                "val/sequence_lengths_solved": float(sequence_length_solved.mean()) if len(sequence_length_solved) > 0 else 0.0,
                "val/stop_rate": stop_rate,
                "val/advantages_mean": float(filtered_advantages.mean()),
                "val/advantages_min": float(filtered_advantages.min()),
                "val/advantages_max": float(filtered_advantages.max()),
            }
            step_metrics.update({k: float(v) for k, v in reward_metrics.items() if isinstance(v, (int, float, np.floating))})
            step_metrics.update(train_metrics)
            self._log_step_metrics(step_metrics, episode=episode)

            if self.args.save_freq > 0 and training_step % self.args.save_freq == 0:
                ckpt_path = self.checkpoint_dir / f"step_{training_step}"
                LOGGER.info("Saving checkpoint at step %d -> %s", training_step, ckpt_path)
                self._save_model(ckpt_path)
                self._cleanup_old_checkpoints()

            if self.eval_dataset is not None and self.eval_freq > 0 and (training_step - 1) % self.eval_freq == 0:
                self._evaluate(training_step=training_step, episode=episode)
            self._maybe_mark_step_memory(training_step, "end")

        self._mark_memory("training_done")
        LOGGER.info("Training complete. Saving final model to %s", self.output_dir)
        self._save_model(self.output_dir)
        self._mark_memory("final_model_saved")

    def close(self) -> None:
        self._mark_memory("close_start")
        try:
            asyncio.run(cleanup_all_llm_judge_clients())
        except Exception as exc:
            LOGGER.warning("LLM judge cleanup failed: %s", exc)

        if getattr(self, "writer", None) is not None:
            self.writer.close()

        if isinstance(getattr(self, "reference_client", None), VLLMReferenceClient):
            self.reference_client.close()
        if isinstance(getattr(self, "reference_process", None), VLLMServerProcess):
            self.reference_process.stop()

        if self.wandb_run is not None:
            try:
                self.wandb_run.finish()
            except Exception:
                pass

        self._mark_memory("close_before_profiler_stop")
        if self.memory_profiler is not None:
            self.memory_profiler.stop()
