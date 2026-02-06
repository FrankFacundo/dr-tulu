# Run vLLM on macOS (Install + Python Launcher)

This guide shows how to install vLLM on macOS and launch a local OpenAI-compatible vLLM instance from Python.

Checked against upstream docs on February 6, 2026:
- vLLM macOS support is experimental and currently CPU-oriented.
- For Apple Silicon CPU, upstream docs say there are no pre-built wheels, so build from source.

## 1) Prerequisites

- macOS (Apple Silicon or Intel)
- Python 3.11+ (recommended: 3.12)
- Xcode Command Line Tools

Install tooling:

```bash
xcode-select --install
```

From `rl/open-instruct`, create a dedicated environment:

```bash
cd rl/open-instruct
python3 -m venv .venv-vllm
source .venv-vllm/bin/activate
export PYTHONNOUSERSITE=1
python -m pip install --upgrade pip wheel packaging ninja "setuptools-scm>=8" "numpy<2.3"
```

## 2) Install vLLM (official source build for macOS CPU)

Clone vLLM and build with CPU target:

```bash
cd /tmp
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install torch
python use_existing_torch.py
pip install -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python -m pip install --no-build-isolation -e .
pip install openai
```

Do not use `python setup.py install` here. It uses legacy `easy_install` and can end with dependency resolution errors on dev versions.

Quick sanity check:

```bash
python -c "import vllm; print(vllm.__version__)"
python -c "import cbor2; print('cbor2 ok')"
python -c "import site; print('ENABLE_USER_SITE =', site.ENABLE_USER_SITE)"
```

## 3) Launch vLLM from Python

This repo now includes:

- `scripts/vllm_mac_launcher.py`

Run it:

```bash
cd /Users/frankfacundo/Code/dr-tulu/rl/open-instruct
source .venv-vllm/bin/activate
python scripts/vllm_mac_launcher.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --dtype float16
```

What it does:
- Starts `vllm serve` on `http://127.0.0.1:8000`
- Detects whether your vLLM CLI supports `--device` and only passes it when available
- Waits until `/v1/models` is healthy
- Sends one chat request through the OpenAI Python SDK
- Prints the response
- Stops the server (unless `--keep-alive` is set)

Keep server alive after the test request:

```bash
python scripts/vllm_mac_launcher.py --keep-alive
```

## 4) Manual serve command (without launcher script)

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype float16 \
  --api-key local-dev-key
```

If your vLLM version supports `--device`, you can add `--device cpu`.

## 5) Notes for this repository

In this codebase, RL rollout backend selection currently requires CUDA for `vllm` mode (`open_instruct/device_utils.py`). On macOS, use `--rollout_backend hf` for training flows such as `open_instruct/grpo_fast.py`.

## Troubleshooting

- First startup can be slow because model weights are downloaded and loaded.
- If memory is tight, use a smaller model and/or lower `--max-model-len`.
- If you hit dtype issues, switch from `--dtype float16` to `--dtype float32`.
- For gated Hugging Face models, run `huggingface-cli login` first.
- If you want Apple GPU acceleration, see the community `vllm-metal` project (separate from official vLLM support).
- If you see `ModuleNotFoundError: No module named 'cbor2'`, run:

```bash
python -m pip install cbor2
```
- If install ends with:
  - `Could not find suitable distribution for Requirement.parse('vllm==...')`
  reinstall with modern pip from the vLLM source directory:

```bash
cd /path/to/vllm
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python -m pip install --no-build-isolation -e .
```
- If you see mixed package paths (for example `vllm` or `torch` importing from `~/.local/...` while inside a conda/venv env), force isolation and reinstall:

```bash
export PYTHONNOUSERSITE=1
python -m pip uninstall -y vllm torch torchaudio torchvision numpy numba
python -m pip install --upgrade pip setuptools wheel "numpy<2.3"
cd /path/to/vllm
python -m pip install -r requirements/cpu-build.txt --extra-index-url https://download.pytorch.org/whl/cpu
VLLM_TARGET_DEVICE=cpu python -m pip install --no-build-isolation -e .
python -m pip check
```

## References

- vLLM installation docs: https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html
- vLLM quickstart (`vllm serve`, OpenAI API usage): https://docs.vllm.ai/en/stable/getting_started/quickstart.html
- vLLM OpenAI-compatible server docs: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- vllm-metal (community project): https://github.com/vllm-project/vllm-metal
