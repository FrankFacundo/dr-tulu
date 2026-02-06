import pytest

from open_instruct.device_utils import (
    resolve_attn_implementation,
    resolve_device_type,
    resolve_dtype_str,
    resolve_rollout_backend,
    resolve_training_backend,
)


def test_resolve_device_type_priority():
    assert resolve_device_type(None, cuda_available=True, mps_available=True) == "cuda"
    assert resolve_device_type(None, cuda_available=False, mps_available=True) == "mps"
    assert resolve_device_type(None, cuda_available=False, mps_available=False) == "cpu"


def test_resolve_dtype_str():
    assert resolve_dtype_str("cuda", cuda_bf16_supported=True) == "bfloat16"
    assert resolve_dtype_str("cuda", cuda_bf16_supported=False) == "float16"
    assert resolve_dtype_str("mps") == "float16"
    assert resolve_dtype_str("cpu") == "float32"


def test_resolve_attn_implementation():
    assert resolve_attn_implementation("cuda") == "flash_attention_2"
    assert resolve_attn_implementation("mps") == "eager"
    assert resolve_attn_implementation("cpu") == "eager"


def test_resolve_training_backend():
    assert resolve_training_backend("auto", device_type="cpu", deepspeed_available=False) == "torch"
    assert resolve_training_backend("auto", device_type="cuda", deepspeed_available=True) == "deepspeed"
    with pytest.raises(ValueError):
        resolve_training_backend("deepspeed", device_type="cpu", deepspeed_available=True)


def test_resolve_rollout_backend():
    assert resolve_rollout_backend("auto", device_type="cpu", vllm_available=False) == "hf"
    assert resolve_rollout_backend("auto", device_type="cuda", vllm_available=True) == "vllm"
    with pytest.raises(ValueError):
        resolve_rollout_backend("vllm", device_type="mps", vllm_available=True)
