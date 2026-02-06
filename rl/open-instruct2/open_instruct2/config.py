from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional

from transformers import HfArgumentParser


@dataclass
class TrainArgs:
    # Experiment
    exp_name: str = "grpo_single_process"
    seed: int = 1
    output_dir: str = "output_open_instruct2"
    run_name: Optional[str] = None
    log_level: str = "INFO"
    smoke_test: bool = False
    """Run a tiny end-to-end training check with lightweight defaults."""
    smoke_test_steps: int = 2
    """Number of training steps to run in smoke test mode."""
    enable_memory_profile: bool = False
    """Enable periodic memory profiling (main process + child processes like local vLLM)."""
    memory_profile_interval_s: float = 1.0
    """Sampling interval in seconds for memory profiler background collection."""
    memory_profile_top_n_children: int = 8
    """How many top child processes by RSS to keep per sample."""
    memory_profile_mark_steps: bool = True
    """Whether to inject explicit memory markers at training step boundaries."""
    memory_profile_step_interval: int = 1
    """How often to mark steps when memory_profile_mark_steps is enabled."""
    memory_profile_enable_tracemalloc: bool = False
    """Track Python heap with tracemalloc (higher overhead, Python objects only)."""

    # Runtime target
    runtime_target: Literal["auto", "linux_gpu", "mac"] = "auto"
    """Preset runtime target. `mac` selects MPS/CPU; `linux_gpu` requires CUDA."""
    device: Optional[str] = None
    """Optional explicit torch device (`cuda`, `mps`, `cpu`, or `cuda:0`)."""
    policy_dtype: Optional[Literal["float32", "float16", "bfloat16"]] = None
    """Optional dtype override for policy model."""
    attn_implementation: Optional[Literal["flash_attention_2", "sdpa", "eager"]] = None
    """Optional attention implementation override."""

    # Model / tokenizer
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    model_revision: str = "main"
    tokenizer_name_or_path: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    trust_remote_code: bool = False
    use_fast_tokenizer: bool = True
    chat_template_name: Optional[str] = None
    add_bos: bool = False
    tokenizer_fn: str = "get_tokenizer_tulu_v2_2"
    gradient_checkpointing: bool = True

    # Training mode
    train_mode: Literal["full", "lora"] = "full"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None

    # Dataset
    dataset_mixer_list: List[str] = field(default_factory=lambda: ["ai2-adapt-dev/rlvr_gsm8k_zs", "1.0"])
    dataset_mixer_eval_list: List[str] = field(default_factory=lambda: [])
    dataset_mixer_list_splits: List[str] = field(default_factory=lambda: ["train"])
    dataset_mixer_eval_list_splits: List[str] = field(default_factory=lambda: ["test"])
    dataset_transform_fn: List[str] = field(
        default_factory=lambda: ["rlvr_tokenize_rl_rag_v1", "rlvr_filter_v1"]
    )
    dataset_cache_mode: Literal["hf", "local"] = "local"
    dataset_local_cache_dir: str = "local_dataset_cache"
    dataset_config_hash: Optional[str] = None
    dataset_config_eval_hash: Optional[str] = None
    dataset_skip_cache: bool = False
    cache_dataset_only: bool = False
    shuffle_eval_dataset: bool = False
    max_token_length: int = 512
    max_prompt_token_length: int = 256
    system_prompt_file: Optional[str] = None

    # Rollout
    total_episodes: int = 100_000
    num_training_steps: Optional[int] = None
    num_unique_prompts_rollout: int = 16
    num_samples_per_prompt_rollout: int = 4
    response_length: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    stop_strings: Optional[List[str]] = None

    # Optimization
    learning_rate: float = 2e-5
    weight_decay: float = 0.0
    lr_scheduler_type: Literal[
        "linear",
        "cosine",
        "cosine_with_restarts",
        "polynomial",
        "constant",
        "constant_with_warmup",
    ] = "linear"
    warm_up_steps: int = 0
    warmup_ratio: float = 0.0
    max_grad_norm: float = 1.0
    per_device_train_batch_size: int = 1
    num_epochs: int = 1
    num_mini_batches: int = 1
    beta: float = 0.05
    clip_lower: float = 0.2
    clip_higher: float = 0.2
    kl_estimator: Literal["kl1", "kl2", "kl3", "kl4"] = "kl3"
    advantage_normalization_type: Literal["standard", "centered"] = "standard"
    mask_truncated_completions: bool = False

    # Rewards
    apply_verifiable_reward: bool = True
    verification_reward: float = 10.0
    overwrite_reward_fn_tag: Optional[str] = None
    apply_r1_style_format_reward: bool = False
    r1_style_format_reward: float = 1.0
    apply_rl_rag_format_reward: bool = False
    additive_format_reward: bool = False
    non_stop_penalty: bool = False
    non_stop_penalty_value: float = 0.0
    only_reward_good_outputs: bool = False

    # Rubric options (mirrors grpo_fast_rubric.py behavior)
    use_general_rubric: bool = False
    evaluate_closed_book_answer: bool = False
    apply_adaptive_rubric_reward: bool = False
    use_full_responses_for_adaptive_rubric: bool = True
    answer_length_limit_in_words: Optional[int] = 450
    normalize_rubric_scores: bool = False
    use_rubric_buffer: bool = False
    max_active_rubrics: int = 5
    use_static_rubrics_as_persistent_rubrics: bool = True
    add_static_rubrics_to_active_rubrics_every_n_steps: int = 10
    use_likert_rubric: bool = False
    no_citation_reward: bool = False
    use_full_response_as_answer: bool = False
    save_adaptive_rubrics: bool = False
    cache_adaptive_rubric_data_dir: Optional[str] = None
    mcp_parser_name: Optional[str] = None
    log_direction_agreement: bool = False

    # Verifier options
    verifier_strategy: str = "judge"
    llm_judge_model: str = "azure/gpt-4o-mini-standard"
    llm_judge_max_tokens: int = 2048
    llm_judge_temperature: float = 1.0
    llm_judge_timeout: int = 60
    llm_judge_max_context_length: int = 2048
    code_api_url: str = os.environ.get("CODE_API_URL", "http://localhost:1234") + "/test_program"
    code_max_execution_time: float = 1.0

    # vLLM reference model
    reference_mode: Literal["launch_local_vllm", "endpoint_vllm"] = "launch_local_vllm"
    """Launch a vLLM server in this script or call an existing endpoint."""
    reference_endpoint: str = "http://127.0.0.1:8000"
    reference_api_key: Optional[str] = None
    reference_model_name_or_path: Optional[str] = None
    reference_launch_host: str = "127.0.0.1"
    reference_launch_port: int = 8000
    reference_launch_dtype: Optional[str] = "auto"
    reference_launch_gpu_memory_utilization: float = 0.9
    reference_launch_tensor_parallel_size: int = 1
    reference_launch_max_model_len: Optional[int] = None
    reference_launch_extra_args: Optional[str] = None
    reference_startup_timeout_s: int = 240
    reference_timeout_s: int = 120
    reference_max_retries: int = 3

    # Eval / checkpointing
    num_evals: int = 10
    eval_max_samples: int = 256
    save_freq: int = -1
    keep_last_n_checkpoints: int = 3
    with_tracking: bool = False
    wandb_project_name: str = "open_instruct2"
    wandb_entity: Optional[str] = None

    def __post_init__(self) -> None:
        if self.num_samples_per_prompt_rollout <= 0:
            raise ValueError("num_samples_per_prompt_rollout must be > 0.")
        if self.num_unique_prompts_rollout <= 0:
            raise ValueError("num_unique_prompts_rollout must be > 0.")
        if self.response_length <= 0:
            raise ValueError("response_length must be > 0.")
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be > 0.")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0.")
        if self.num_mini_batches <= 0:
            raise ValueError("num_mini_batches must be > 0.")
        if self.reference_mode == "endpoint_vllm" and not self.reference_endpoint:
            raise ValueError("reference_endpoint must be set when reference_mode=endpoint_vllm.")
        if self.train_mode == "lora" and self.lora_r <= 0:
            raise ValueError("lora_r must be > 0 when train_mode=lora.")
        if self.memory_profile_interval_s <= 0:
            raise ValueError("memory_profile_interval_s must be > 0.")
        if self.memory_profile_top_n_children <= 0:
            raise ValueError("memory_profile_top_n_children must be > 0.")
        if self.memory_profile_step_interval <= 0:
            raise ValueError("memory_profile_step_interval must be > 0.")
        if not (
            self.apply_verifiable_reward
            or self.apply_r1_style_format_reward
            or self.apply_rl_rag_format_reward
            or self.non_stop_penalty
        ):
            raise ValueError(
                "At least one reward path must be active: verifiable, format reward, or non-stop penalty."
            )


def parse_args() -> TrainArgs:
    parser = HfArgumentParser(TrainArgs)
    (args,) = parser.parse_args_into_dataclasses()
    return args


def apply_smoke_test_overrides(args: TrainArgs) -> TrainArgs:
    if not args.smoke_test:
        return args

    # Keep smoke runs deterministic and cheap.
    args.with_tracking = False
    args.save_adaptive_rubrics = False
    args.cache_adaptive_rubric_data_dir = None
    args.apply_adaptive_rubric_reward = False
    args.use_rubric_buffer = False
    args.normalize_rubric_scores = False

    args.num_unique_prompts_rollout = 1
    args.num_samples_per_prompt_rollout = 2
    args.per_device_train_batch_size = 1
    args.num_epochs = 1
    args.num_mini_batches = 1
    args.response_length = min(args.response_length, 64)
    args.max_prompt_token_length = min(args.max_prompt_token_length, 256)

    args.num_training_steps = max(1, args.smoke_test_steps)
    args.total_episodes = args.num_training_steps * args.num_unique_prompts_rollout * args.num_samples_per_prompt_rollout
    args.eval_max_samples = min(args.eval_max_samples, 8)
    args.num_evals = 1 if len(args.dataset_mixer_eval_list) > 0 else 0
    args.save_freq = -1

    return args
