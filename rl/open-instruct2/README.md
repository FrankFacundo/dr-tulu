# Open-Instruct2

`open-instruct2` is a clean, single-process GRPO training project inspired by `open_instruct/grpo_fast.py`, but designed for easier single-machine research workflows:

- No Ray
- No DeepSpeed
- No multi-node / multi-GPU orchestration
- Policy training in PyTorch + Transformers (`AutoModelForCausalLM`)
- Reference model scoring via vLLM (local launched server or external endpoint)
- Rubric reward support (including adaptive rubrics + rubric buffer options)
- Optional LoRA or full-model training
- Runtime target switch for Linux GPU vs Mac

## Project Layout

```text
rl/open-instruct2/
  open_instruct2/
    bootstrap.py
    config.py
    reference_vllm.py
    rewards.py
    runtime.py
    trainer.py
    train_grpo.py
  pyproject.toml
  README.md
```

## Install

From repo root:

```bash
cd rl/open-instruct2
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

This project reuses verifier and dataset utilities from the sibling `rl/open-instruct` package. If you have not installed it yet:

```bash
pip install -e ../open-instruct
```

## Runtime Modes

Set `--runtime_target` to one of:

- `linux_gpu`: requires CUDA.
- `mac`: uses MPS if available, otherwise CPU.
- `auto`: CUDA -> MPS -> CPU fallback.

## vLLM Reference Modes

Set `--reference_mode` to one of:

- `launch_local_vllm`: launches `vllm.entrypoints.openai.api_server` inside this script.
- `endpoint_vllm`: calls an already-running vLLM OpenAI-compatible endpoint.

On Mac, you can use `launch_local_vllm` if vLLM is installed in the same Python environment.

## Launch Examples

### 1) Linux GPU, full fine-tuning, launch local vLLM in-script

```bash
cd rl/open-instruct2
source .venv/bin/activate

python -m open_instruct2.train_grpo \
  --runtime_target linux_gpu \
  --train_mode full \
  --reference_mode launch_local_vllm \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train \
  --total_episodes 256 \
  --num_unique_prompts_rollout 8 \
  --num_samples_per_prompt_rollout 2 \
  --per_device_train_batch_size 2 \
  --response_length 256 \
  --apply_verifiable_reward true
```

### 2) Linux GPU, LoRA training, launch local vLLM in-script

```bash
python -m open_instruct2.train_grpo \
  --runtime_target linux_gpu \
  --train_mode lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --reference_mode launch_local_vllm \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train \
  --total_episodes 256
```

### 3) Linux GPU, external vLLM endpoint

Start endpoint separately:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000
```

Then train:

```bash
python -m open_instruct2.train_grpo \
  --runtime_target linux_gpu \
  --reference_mode endpoint_vllm \
  --reference_endpoint http://127.0.0.1:8000 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train \
  --total_episodes 256
```

### 4) Mac (MPS/CPU), launch local vLLM in-script

```bash
python -m open_instruct2.train_grpo \
  --runtime_target mac \
  --reference_mode launch_local_vllm \
  --reference_launch_host 127.0.0.1 \
  --reference_launch_port 8000 \
  --reference_api_key local-dev-key \
  --reference_launch_dtype float16 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train \
  --total_episodes 128
```

### 5) Mac (MPS/CPU) with external vLLM endpoint (fallback)

Run vLLM on a Linux GPU host, then:

```bash
python -m open_instruct2.train_grpo \
  --runtime_target mac \
  --reference_mode endpoint_vllm \
  --reference_endpoint http://<linux-gpu-host>:8000 \
  --train_mode lora \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train \
  --total_episodes 128
```

## Smoke Test

Use `--smoke_test true` to run a minimal end-to-end validation run. This mode automatically reduces training to a tiny configuration:

- 2 training steps (configurable with `--smoke_test_steps`)
- 1 prompt x 2 samples per rollout
- short generation length
- no wandb/adaptive-rubric heavy paths

### Linux GPU smoke test (launch local vLLM)

```bash
python -m open_instruct2.train_grpo \
  --smoke_test true \
  --runtime_target linux_gpu \
  --reference_mode launch_local_vllm \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train
```

### Mac smoke test (launch local vLLM)

```bash
python -m open_instruct2.train_grpo \
  --smoke_test true \
  --runtime_target mac \
  --reference_mode launch_local_vllm \
  --reference_launch_host 127.0.0.1 \
  --reference_launch_port 8000 \
  --reference_api_key local-dev-key \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train
```

### Mac smoke test (external vLLM endpoint fallback)

```bash
python -m open_instruct2.train_grpo \
  --smoke_test true \
  --runtime_target mac \
  --reference_mode endpoint_vllm \
  --reference_endpoint http://<linux-gpu-host>:8000 \
  --model_name_or_path Qwen/Qwen3-0.6B \
  --dataset_mixer_list ai2-adapt-dev/rlvr_gsm8k_zs 1.0 \
  --dataset_mixer_list_splits train
```

## Rubrics

Rubric options from `grpo_fast_rubric.py` are available, including:

- `--use_general_rubric`
- `--apply_adaptive_rubric_reward`
- `--use_rubric_buffer`
- `--normalize_rubric_scores`
- `--max_active_rubrics`
- `--use_static_rubrics_as_persistent_rubrics`

If rubric/adaptive scoring uses LLM judges, configure credentials (for your verifier stack), e.g.:

```bash
export OPENAI_API_KEY=...
```

## Logging and Outputs

- TensorBoard logs: `output_open_instruct2/<run_name>/tensorboard`
- Checkpoints: `output_open_instruct2/<run_name>/checkpoints/step_*`
- Final model: `output_open_instruct2/<run_name>/`
- Optional adaptive rubric dumps: `adaptive_rubrics_<run_name>.jsonl`

## Notes

- `stop_strings` are used for finish-reason detection only (generation hard-stop by arbitrary string is not enforced in this first clean version).
- This trainer is intentionally single-process for reliability and readability.
- For RAM profiling (including child vLLM processes), see `README_memory_profiling.md`.
