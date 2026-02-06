from __future__ import annotations

from typing import Optional

import torch


def resolve_device_type(preferred: Optional[str], cuda_available: bool, mps_available: bool) -> str:
    if preferred:
        return preferred.split(":")[0]
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    device_type = resolve_device_type(
        preferred,
        torch.cuda.is_available(),
        torch.backends.mps.is_available(),
    )
    return torch.device(device_type)


def resolve_dtype_str(device_type: str, cuda_bf16_supported: Optional[bool] = None) -> str:
    if device_type == "cuda":
        if cuda_bf16_supported is None:
            cuda_bf16_supported = torch.cuda.is_bf16_supported()
        return "bfloat16" if cuda_bf16_supported else "float16"
    if device_type == "mps":
        return "float16"
    return "float32"


def torch_dtype_from_str(dtype_str: str) -> torch.dtype:
    return getattr(torch, dtype_str)


def resolve_attn_implementation(device_type: str, override: Optional[str] = None) -> str:
    if override:
        return override
    if device_type == "cuda":
        return "flash_attention_2"
    return "eager"


def resolve_training_backend(requested: str, device_type: str, deepspeed_available: bool) -> str:
    if requested == "auto":
        if device_type == "cuda" and deepspeed_available:
            return "deepspeed"
        return "torch"
    if requested == "deepspeed":
        if device_type != "cuda":
            raise ValueError("DeepSpeed backend requires CUDA; use --training_backend torch on non-CUDA devices.")
        if not deepspeed_available:
            raise ValueError("DeepSpeed is not available; install it or use --training_backend torch.")
        return "deepspeed"
    if requested == "torch":
        return "torch"
    raise ValueError(f"Unknown training backend: {requested}")


def resolve_rollout_backend(requested: str, device_type: str, vllm_available: bool) -> str:
    if requested == "auto":
        if device_type == "cuda" and vllm_available:
            return "vllm"
        return "hf"
    if requested == "vllm":
        if device_type != "cuda":
            raise ValueError("vLLM backend requires CUDA; use --rollout_backend hf on non-CUDA devices.")
        if not vllm_available:
            raise ValueError("vLLM is not available; install it or use --rollout_backend hf.")
        return "vllm"
    if requested == "hf":
        return "hf"
    raise ValueError(f"Unknown rollout backend: {requested}")


def maybe_empty_cache(device_type: str) -> None:
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        return
    if device_type == "mps" and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
