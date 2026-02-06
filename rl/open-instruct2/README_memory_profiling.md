# Open-Instruct2 Memory Profiling (Mac + local vLLM)

This guide profiles RAM usage for:

- the training process (`open_instruct2.train_grpo`)
- child processes (including local vLLM server + engine workers)
- optional Python heap (`tracemalloc`)

## 1) Install profiling dependency

From `rl/open-instruct2`:

```bash
pip install -e .
```

(`psutil` is now part of `open-instruct2` dependencies.)

## 2) Run your command with memory profiling enabled

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
  --total_episodes 128 \
  --enable_memory_profile true \
  --memory_profile_interval_s 1.0 \
  --memory_profile_mark_steps true \
  --memory_profile_step_interval 1
```

Optional (more detail, higher overhead):

```bash
--memory_profile_enable_tracemalloc true
```

## 3) Output files

Memory profile artifacts are written to:

```text
output_open_instruct2/<run_name>/memory_profile/
```

Files:

- `memory_samples.csv`: time-series numeric metrics
- `memory_samples.jsonl`: per-sample details, including top child processes
- `memory_summary.json`: peak values and top child peaks
- `memory_summary.md`: human-readable summary

## 4) How to identify what is using RAM

Open `memory_summary.md` first.

- If `Peak children total RSS` is much larger than `Peak main process RSS`, local vLLM is the main RAM consumer.
- If `Peak main process RSS` is dominant, the trainer/dataset/model process is the main consumer.
- In `Top Child Processes by Peak RSS`, check command lines to distinguish:
  - vLLM API server process
  - vLLM engine/core worker process
  - other subprocesses

For step-level correlation, use markers in `memory_samples.csv` / `memory_samples.jsonl`:

- `step_<N>_start`
- `step_<N>_after_rollout`
- `step_<N>_after_rewards`
- `step_<N>_after_train`
- `step_<N>_end`

## 5) Useful knobs when vLLM on Mac uses too much RAM

- Lower model/context:
  - smaller model
  - lower `--reference_launch_max_model_len`
- Reduce KV cache space used by vLLM CPU backend:

```bash
export VLLM_CPU_KVCACHE_SPACE=16
```

Then rerun and compare peaks in `memory_summary.md`.

## 6) Troubleshooting

- If logs show `Memory profiler cannot enumerate child processes`, your environment is blocking process enumeration.
- In that case, profiling still works for the main training process, but child-process attribution (vLLM engine/server split) will be limited.
