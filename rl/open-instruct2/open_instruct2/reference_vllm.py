from __future__ import annotations

import json
import logging
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import requests
import torch
from transformers import PreTrainedTokenizer

from open_instruct2.config import TrainArgs
from open_instruct2.runtime import RuntimeInfo

LOGGER = logging.getLogger(__name__)


@dataclass
class VLLMServerProcess:
    process: subprocess.Popen
    base_url: str
    stdout_path: Path
    stderr_path: Path

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


class VLLMReferenceClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        tokenizer: PreTrainedTokenizer,
        timeout_s: int,
        max_retries: int,
        api_key: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
        self.base_url = normalized
        self.completions_url = f"{self.base_url}/completions"
        self.health_url = self.base_url.removesuffix("/v1") + "/health"

        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def close(self) -> None:
        self.session.close()

    def check_health(self) -> bool:
        try:
            response = self.session.get(self.health_url, timeout=self.timeout_s)
            return response.status_code == 200
        except Exception:
            return False

    def score_sequences(self, sequences: Sequence[Sequence[int]]) -> List[torch.Tensor]:
        return [self.score_sequence(list(tokens)) for tokens in sequences]

    def score_sequence(self, token_ids: List[int]) -> torch.Tensor:
        if len(token_ids) < 2:
            return torch.zeros(0, dtype=torch.float32)

        payload_templates = [
            {
                "model": self.model_name,
                "prompt": token_ids,
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            },
            {
                "model": self.model_name,
                "prompt": [token_ids],
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            },
        ]
        payload_templates.append(
            {
                "model": self.model_name,
                "prompt": self.tokenizer.decode(token_ids, clean_up_tokenization_spaces=False),
                "max_tokens": 0,
                "temperature": 0.0,
                "echo": True,
                "logprobs": 1,
            }
        )

        last_error: Optional[Exception] = None
        for payload in payload_templates:
            try:
                token_logprobs = self._request_token_logprobs(payload)
                if len(token_logprobs) < len(token_ids):
                    # String fallback can mismatch tokenizer boundaries; if so, reject and try next format.
                    raise ValueError(
                        f"Received {len(token_logprobs)} token logprobs for {len(token_ids)} tokens."
                    )
                if len(token_logprobs) > len(token_ids):
                    token_logprobs = token_logprobs[-len(token_ids) :]
                response_token_logprobs = token_logprobs[1:len(token_ids)]
                return torch.tensor(response_token_logprobs, dtype=torch.float32)
            except Exception as exc:
                last_error = exc
                continue

        raise RuntimeError(f"Failed to score sequence with vLLM endpoint: {last_error}") from last_error

    def _request_token_logprobs(self, payload: dict) -> List[float]:
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.post(
                    self.completions_url,
                    data=json.dumps(payload),
                    timeout=self.timeout_s,
                )
                if response.status_code != 200:
                    raise RuntimeError(f"vLLM response status {response.status_code}: {response.text}")

                data = response.json()
                choices = data.get("choices")
                if not choices:
                    raise RuntimeError(f"No `choices` in vLLM response: {data}")
                choice = choices[0]
                logprobs_dict = choice.get("logprobs", {})
                token_logprobs = logprobs_dict.get("token_logprobs")
                if token_logprobs is None:
                    raise RuntimeError(f"No token logprobs in vLLM response: {data}")
                return [0.0 if lp is None else float(lp) for lp in token_logprobs]
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(backoff)
                backoff *= 2.0
        raise RuntimeError("Unreachable retry loop in _request_token_logprobs.")


def _wait_for_server_health(
    session: requests.Session,
    health_url: str,
    timeout_s: int,
    process: subprocess.Popen,
) -> None:
    start_time = time.time()
    while time.time() - start_time < timeout_s:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM server exited early with code {process.returncode}.")
        try:
            response = session.get(health_url, timeout=2)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for vLLM server health at {health_url}.")


def launch_local_vllm_server(
    args: TrainArgs,
    runtime: RuntimeInfo,
    output_dir: str,
    model_name: str,
) -> VLLMServerProcess:
    logs_dir = Path(output_dir) / "reference_vllm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = logs_dir / "vllm_stdout.log"
    stderr_path = logs_dir / "vllm_stderr.log"

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_name,
        "--host",
        args.reference_launch_host,
        "--port",
        str(args.reference_launch_port),
    ]
    if runtime.device_type == "cuda":
        cmd.extend(
            [
                "--tensor-parallel-size",
                str(args.reference_launch_tensor_parallel_size),
                "--gpu-memory-utilization",
                str(args.reference_launch_gpu_memory_utilization),
            ]
        )
    else:
        LOGGER.info(
            "Launching local vLLM on non-CUDA runtime (%s). "
            "Ensure your vLLM install supports this backend (e.g., macOS CPU build).",
            runtime.device_type,
        )
    if args.reference_launch_dtype:
        cmd.extend(["--dtype", args.reference_launch_dtype])
    if args.reference_launch_max_model_len:
        cmd.extend(["--max-model-len", str(args.reference_launch_max_model_len)])
    if args.reference_launch_extra_args:
        cmd.extend(shlex.split(args.reference_launch_extra_args))

    LOGGER.info("Launching local vLLM reference server: %s", " ".join(cmd))
    stdout_file = stdout_path.open("w")
    stderr_file = stderr_path.open("w")
    process = subprocess.Popen(cmd, stdout=stdout_file, stderr=stderr_file)

    host_for_client = args.reference_launch_host
    if host_for_client in {"0.0.0.0", "::"}:
        host_for_client = "127.0.0.1"
    base_url = f"http://{host_for_client}:{args.reference_launch_port}"
    health_url = f"{base_url}/health"
    session = requests.Session()
    try:
        _wait_for_server_health(session, health_url, args.reference_startup_timeout_s, process)
    except Exception:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=5)
        raise
    finally:
        session.close()
        stdout_file.close()
        stderr_file.close()

    LOGGER.info("Local vLLM reference server is healthy at %s", base_url)
    return VLLMServerProcess(
        process=process,
        base_url=base_url,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def create_reference_client(
    args: TrainArgs,
    runtime: RuntimeInfo,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
) -> Tuple[VLLMReferenceClient, Optional[VLLMServerProcess]]:
    model_name = args.reference_model_name_or_path or args.model_name_or_path
    server_process: Optional[VLLMServerProcess] = None
    endpoint = args.reference_endpoint

    if args.reference_mode == "launch_local_vllm":
        server_process = launch_local_vllm_server(args, runtime, output_dir, model_name=model_name)
        endpoint = server_process.base_url
    elif args.reference_mode != "endpoint_vllm":
        raise ValueError(f"Unsupported reference_mode: {args.reference_mode}")

    client = VLLMReferenceClient(
        base_url=endpoint,
        model_name=model_name,
        tokenizer=tokenizer,
        timeout_s=args.reference_timeout_s,
        max_retries=args.reference_max_retries,
        api_key=args.reference_api_key,
    )
    if not client.check_health():
        raise RuntimeError(
            f"vLLM reference endpoint health check failed at {client.health_url}. "
            "Ensure the endpoint is running and reachable."
        )
    return client, server_process
