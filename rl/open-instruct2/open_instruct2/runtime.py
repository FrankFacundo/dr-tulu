from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from open_instruct2.config import TrainArgs


@dataclass
class RuntimeInfo:
    device: torch.device
    device_type: str
    dtype_str: str
    torch_dtype: torch.dtype
    attn_implementation: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device_type(runtime_target: str, device_override: Optional[str]) -> str:
    if device_override:
        return device_override.split(":")[0]

    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()

    if runtime_target == "linux_gpu":
        if not cuda_available:
            raise RuntimeError("runtime_target=linux_gpu requires CUDA, but CUDA is unavailable.")
        return "cuda"
    if runtime_target == "mac":
        return "mps" if mps_available else "cpu"
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def _resolve_dtype_str(device_type: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    if device_type == "cuda":
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    if device_type == "mps":
        # float32 is safer for training portability on Apple Silicon.
        return "float32"
    return "float32"


def _resolve_attn_impl(device_type: str, override: Optional[str]) -> str:
    if override is not None:
        return override
    if device_type == "cuda":
        return "flash_attention_2"
    return "eager"


def resolve_runtime(args: TrainArgs) -> RuntimeInfo:
    device_type = _resolve_device_type(args.runtime_target, args.device)
    device = torch.device(args.device if args.device else device_type)
    dtype_str = _resolve_dtype_str(device_type, args.policy_dtype)
    torch_dtype = getattr(torch, dtype_str)
    attn_implementation = _resolve_attn_impl(device_type, args.attn_implementation)
    return RuntimeInfo(
        device=device,
        device_type=device_type,
        dtype_str=dtype_str,
        torch_dtype=torch_dtype,
        attn_implementation=attn_implementation,
    )
