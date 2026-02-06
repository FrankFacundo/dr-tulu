#!/usr/bin/env python3
"""Launch a local vLLM server on macOS CPU, send one test request, then stop."""

from __future__ import annotations

import argparse
import importlib.util
import re
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


def resolve_vllm_cli_cmd() -> list[str]:
    env_local = Path(sys.executable).resolve().parent / "vllm"
    if env_local.exists():
        return [str(env_local)]
    discovered = shutil.which("vllm")
    if discovered:
        return [discovered]
    if importlib.util.find_spec("vllm.entrypoints.cli.main") is not None:
        return [sys.executable, "-m", "vllm.entrypoints.cli.main"]
    raise SystemExit(
        "Could not find vLLM CLI in this environment. Install vLLM first (or ensure `vllm` is on PATH)."
    )


def get_serve_supported_flags(vllm_cli_cmd: list[str]) -> set[str]:
    commands_to_try = [
        [*vllm_cli_cmd, "serve", "--help=all"],
        [*vllm_cli_cmd, "serve", "--help"],
    ]
    help_text = ""
    for cmd in commands_to_try:
        try:
            result = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=45,
            )
            help_text = f"{result.stdout}\n{result.stderr}"
            if "usage:" in help_text:
                break
        except Exception:
            continue
    if not help_text:
        return set()
    return set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))


def wait_for_server(base_url: str, timeout_sec: int, process: subprocess.Popen, api_key: str) -> None:
    deadline = time.time() + timeout_sec
    last_error = "no response yet"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited early with code {process.returncode}. "
                "See logs above for the import/runtime error."
            )
        try:
            request = Request(
                f"{base_url}/models",
                method="GET",
                headers={
                    "Authorization": f"Bearer {api_key}",
                },
            )
            with urlopen(request, timeout=5) as response:
                if response.status == 200:
                    return
                last_error = f"http status {response.status}"
        except URLError as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(
        f"vLLM server did not become ready within {timeout_sec} seconds. Last error: {last_error}"
    )


def stop_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start vLLM on macOS CPU, run one chat request, and optionally keep the server running."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct", help="Hugging Face model id.")
    parser.add_argument("--host", default="127.0.0.1", help="Host for the local vLLM server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the local vLLM server.")
    parser.add_argument("--api-key", default="local-dev-key", help="API key required by the vLLM server.")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="dtype for macOS CPU backend (float16 saves memory).",
    )
    parser.add_argument("--max-model-len", type=int, default=2048, help="Maximum context length for the server.")
    parser.add_argument("--timeout-sec", type=int, default=900, help="How long to wait for server readiness.")
    parser.add_argument(
        "--prompt",
        default="Explain in 2 short sentences what vLLM is.",
        help="Prompt sent after server starts.",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the vLLM process alive after the sample request.",
    )
    parser.add_argument(
        "--extra-serve-args",
        nargs="*",
        default=[],
        help="Extra arguments appended to `vllm serve`.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}/v1"
    if importlib.util.find_spec("cbor2") is None:
        raise SystemExit("Missing dependency `cbor2`. Install it with: `python -m pip install cbor2`")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency: install `openai` in this environment (`pip install openai`).") from exc

    vllm_cli_cmd = resolve_vllm_cli_cmd()
    supported_flags = get_serve_supported_flags(vllm_cli_cmd)

    serve_cmd = [
        *vllm_cli_cmd,
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--api-key",
        args.api_key,
        *args.extra_serve_args,
    ]
    if "--device" in supported_flags:
        serve_cmd.extend(["--device", "cpu"])
    else:
        print("Note: this vLLM CLI does not support `--device`; starting with its default backend.")

    print("Starting vLLM server:")
    print(" ".join(serve_cmd))
    process = subprocess.Popen(serve_cmd)

    try:
        wait_for_server(base_url, args.timeout_sec, process, args.api_key)
        print(f"vLLM is ready at {base_url}")

        client = OpenAI(base_url=base_url, api_key=args.api_key)
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": args.prompt},
            ],
            temperature=0.2,
            max_tokens=128,
        )
        print("\nModel response:\n")
        print(response.choices[0].message.content or "")

        if args.keep_alive:
            print("\nServer is still running. Press Ctrl+C to stop it.")
            process.wait()
    except KeyboardInterrupt:
        print("\nInterrupted. Stopping server...")
    finally:
        stop_process(process)
    return 0


if __name__ == "__main__":
    sys.exit(main())
