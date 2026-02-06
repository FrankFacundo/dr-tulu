from __future__ import annotations

import csv
import json
import logging
import os
import threading
import time
import tracemalloc
from pathlib import Path
from typing import Dict, List, Optional

import psutil

LOGGER = logging.getLogger(__name__)
_GB = float(1024**3)
_MB = float(1024**2)


def _bytes_to_gb(num_bytes: float) -> float:
    return float(num_bytes) / _GB


def _sanitize_cmdline(parts: List[str]) -> str:
    masked_flags = {"--api-key", "--hf-token"}
    out: List[str] = []
    i = 0
    while i < len(parts):
        part = str(parts[i])
        out.append(part)
        if part in masked_flags and i + 1 < len(parts):
            out.append("***")
            i += 2
            continue
        i += 1
    return " ".join(out)


class MemoryProfiler:
    def __init__(
        self,
        output_dir: Path,
        interval_s: float = 1.0,
        top_n_children: int = 8,
        enable_tracemalloc: bool = False,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.output_dir = output_dir
        self.interval_s = interval_s
        self.top_n_children = top_n_children
        self.enable_tracemalloc = enable_tracemalloc
        self.logger = logger or LOGGER

        self._proc = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._started = False
        self._start_monotonic = 0.0

        self._csv_file = None
        self._jsonl_file = None
        self._csv_writer = None

        self._latest_snapshot: Optional[Dict] = None
        self._samples: List[Dict] = []
        self._child_peaks: Dict[str, Dict] = {}
        self._children_scan_failed_once = False

    def start(self) -> None:
        if self._started:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.output_dir / "memory_samples.csv"
        jsonl_path = self.output_dir / "memory_samples.jsonl"

        self._csv_file = csv_path.open("w", newline="", encoding="utf-8")
        self._jsonl_file = jsonl_path.open("w", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "timestamp_iso",
                "elapsed_s",
                "label",
                "main_rss_gb",
                "main_vms_gb",
                "children_rss_gb",
                "total_tracked_rss_gb",
                "system_used_gb",
                "system_available_gb",
                "system_percent",
                "python_heap_mb",
                "python_heap_peak_mb",
                "torch_cuda_alloc_gb",
                "torch_cuda_reserved_gb",
                "torch_mps_alloc_gb",
                "torch_mps_driver_gb",
            ],
        )
        self._csv_writer.writeheader()
        self._csv_file.flush()

        if self.enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start(25)

        self._start_monotonic = time.monotonic()
        self._started = True
        self.mark("profiler_start")

        self._thread = threading.Thread(target=self._run_loop, name="memory-profiler", daemon=True)
        self._thread.start()
        self.logger.info("Memory profiler started. Artifacts: %s", str(self.output_dir))

    def stop(self) -> None:
        if not self._started:
            return

        self.mark("profiler_stop_requested")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.mark("profiler_stopped")

        self._write_summary_files()

        if self._csv_file is not None:
            self._csv_file.close()
        if self._jsonl_file is not None:
            self._jsonl_file.close()

        if self.enable_tracemalloc and tracemalloc.is_tracing():
            tracemalloc.stop()

        self._started = False
        self.logger.info("Memory profiler stopped.")

    def mark(self, label: str) -> None:
        if not self._started:
            return
        snapshot = self._collect_snapshot(label=label)
        with self._lock:
            self._latest_snapshot = snapshot
            self._samples.append(snapshot)
            self._write_snapshot(snapshot)

    def latest_metrics_gb(self) -> Dict[str, float]:
        with self._lock:
            if self._latest_snapshot is None:
                return {}
            s = self._latest_snapshot

        metrics: Dict[str, float] = {
            "mem/main_rss_gb": _bytes_to_gb(s["main_rss_bytes"]),
            "mem/children_rss_gb": _bytes_to_gb(s["children_rss_bytes"]),
            "mem/total_rss_gb": _bytes_to_gb(s["total_tracked_rss_bytes"]),
            "mem/system_used_gb": _bytes_to_gb(s["system_used_bytes"]),
            "mem/system_percent": float(s["system_percent"]),
        }
        if s["python_heap_bytes"] is not None:
            metrics["mem/python_heap_mb"] = float(s["python_heap_bytes"]) / _MB
            metrics["mem/python_heap_peak_mb"] = float(s["python_heap_peak_bytes"]) / _MB
        if s["torch_cuda_alloc_bytes"] is not None:
            metrics["mem/torch_cuda_alloc_gb"] = _bytes_to_gb(s["torch_cuda_alloc_bytes"])
        if s["torch_cuda_reserved_bytes"] is not None:
            metrics["mem/torch_cuda_reserved_gb"] = _bytes_to_gb(s["torch_cuda_reserved_bytes"])
        if s["torch_mps_alloc_bytes"] is not None:
            metrics["mem/torch_mps_alloc_gb"] = _bytes_to_gb(s["torch_mps_alloc_bytes"])
        if s["torch_mps_driver_bytes"] is not None:
            metrics["mem/torch_mps_driver_gb"] = _bytes_to_gb(s["torch_mps_driver_bytes"])
        return metrics

    def _run_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            try:
                self.mark("interval")
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Memory profiler sample failed: %s", exc)

    def _collect_snapshot(self, label: str) -> Dict:
        now = time.time()
        elapsed_s = time.monotonic() - self._start_monotonic

        main_info = self._proc.memory_info()
        main_rss_bytes = int(main_info.rss)
        main_vms_bytes = int(main_info.vms)

        children = []
        children_rss_bytes = 0
        try:
            child_processes = self._proc.children(recursive=True)
        except Exception as exc:
            child_processes = []
            if not self._children_scan_failed_once:
                self._children_scan_failed_once = True
                self.logger.warning("Memory profiler cannot enumerate child processes: %s", exc)

        for child in child_processes:
            try:
                with child.oneshot():
                    child_rss = int(child.memory_info().rss)
                    if child_rss <= 0:
                        continue
                    child_pid = int(child.pid)
                    child_name = child.name()
                    child_cmdline = _sanitize_cmdline(child.cmdline())
                    child_create_time = float(child.create_time())
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

            children_rss_bytes += child_rss
            child_record = {
                "pid": child_pid,
                "name": child_name,
                "cmdline": child_cmdline,
                "rss_bytes": child_rss,
            }
            children.append(child_record)

            child_key = f"{child_pid}:{child_create_time}"
            peak = self._child_peaks.get(child_key)
            if peak is None or child_rss > peak["peak_rss_bytes"]:
                self._child_peaks[child_key] = {
                    "pid": child_pid,
                    "name": child_name,
                    "cmdline": child_cmdline,
                    "peak_rss_bytes": child_rss,
                }

        children.sort(key=lambda item: item["rss_bytes"], reverse=True)
        top_children = children[: self.top_n_children]

        vm = psutil.virtual_memory()
        python_heap_bytes = None
        python_heap_peak_bytes = None
        if self.enable_tracemalloc and tracemalloc.is_tracing():
            python_heap_bytes, python_heap_peak_bytes = tracemalloc.get_traced_memory()

        torch_stats = self._collect_torch_stats()

        return {
            "timestamp": now,
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "elapsed_s": elapsed_s,
            "label": label,
            "main_rss_bytes": main_rss_bytes,
            "main_vms_bytes": main_vms_bytes,
            "children_rss_bytes": children_rss_bytes,
            "total_tracked_rss_bytes": main_rss_bytes + children_rss_bytes,
            "system_used_bytes": int(vm.used),
            "system_available_bytes": int(vm.available),
            "system_percent": float(vm.percent),
            "python_heap_bytes": python_heap_bytes,
            "python_heap_peak_bytes": python_heap_peak_bytes,
            "torch_cuda_alloc_bytes": torch_stats["cuda_alloc_bytes"],
            "torch_cuda_reserved_bytes": torch_stats["cuda_reserved_bytes"],
            "torch_mps_alloc_bytes": torch_stats["mps_alloc_bytes"],
            "torch_mps_driver_bytes": torch_stats["mps_driver_bytes"],
            "top_children": top_children,
        }

    def _collect_torch_stats(self) -> Dict[str, Optional[int]]:
        stats: Dict[str, Optional[int]] = {
            "cuda_alloc_bytes": None,
            "cuda_reserved_bytes": None,
            "mps_alloc_bytes": None,
            "mps_driver_bytes": None,
        }
        try:
            import torch
        except Exception:
            return stats

        try:
            if torch.cuda.is_available():
                stats["cuda_alloc_bytes"] = int(torch.cuda.memory_allocated())
                stats["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved())
        except Exception:
            pass

        try:
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                if hasattr(torch.mps, "current_allocated_memory"):
                    stats["mps_alloc_bytes"] = int(torch.mps.current_allocated_memory())
                if hasattr(torch.mps, "driver_allocated_memory"):
                    stats["mps_driver_bytes"] = int(torch.mps.driver_allocated_memory())
        except Exception:
            pass

        return stats

    def _write_snapshot(self, snapshot: Dict) -> None:
        if self._csv_writer is None or self._csv_file is None or self._jsonl_file is None:
            return
        row = {
            "timestamp_iso": snapshot["timestamp_iso"],
            "elapsed_s": round(snapshot["elapsed_s"], 3),
            "label": snapshot["label"],
            "main_rss_gb": round(_bytes_to_gb(snapshot["main_rss_bytes"]), 4),
            "main_vms_gb": round(_bytes_to_gb(snapshot["main_vms_bytes"]), 4),
            "children_rss_gb": round(_bytes_to_gb(snapshot["children_rss_bytes"]), 4),
            "total_tracked_rss_gb": round(_bytes_to_gb(snapshot["total_tracked_rss_bytes"]), 4),
            "system_used_gb": round(_bytes_to_gb(snapshot["system_used_bytes"]), 4),
            "system_available_gb": round(_bytes_to_gb(snapshot["system_available_bytes"]), 4),
            "system_percent": round(float(snapshot["system_percent"]), 2),
            "python_heap_mb": (
                None if snapshot["python_heap_bytes"] is None else round(snapshot["python_heap_bytes"] / _MB, 2)
            ),
            "python_heap_peak_mb": (
                None
                if snapshot["python_heap_peak_bytes"] is None
                else round(snapshot["python_heap_peak_bytes"] / _MB, 2)
            ),
            "torch_cuda_alloc_gb": (
                None
                if snapshot["torch_cuda_alloc_bytes"] is None
                else round(_bytes_to_gb(snapshot["torch_cuda_alloc_bytes"]), 4)
            ),
            "torch_cuda_reserved_gb": (
                None
                if snapshot["torch_cuda_reserved_bytes"] is None
                else round(_bytes_to_gb(snapshot["torch_cuda_reserved_bytes"]), 4)
            ),
            "torch_mps_alloc_gb": (
                None
                if snapshot["torch_mps_alloc_bytes"] is None
                else round(_bytes_to_gb(snapshot["torch_mps_alloc_bytes"]), 4)
            ),
            "torch_mps_driver_gb": (
                None
                if snapshot["torch_mps_driver_bytes"] is None
                else round(_bytes_to_gb(snapshot["torch_mps_driver_bytes"]), 4)
            ),
        }
        self._csv_writer.writerow(row)
        self._csv_file.flush()
        self._jsonl_file.write(json.dumps(snapshot) + "\n")
        self._jsonl_file.flush()

    def _write_summary_files(self) -> None:
        with self._lock:
            if len(self._samples) == 0:
                return
            samples = list(self._samples)
            child_peaks = list(self._child_peaks.values())

        peak_main = max(samples, key=lambda item: item["main_rss_bytes"])
        peak_children = max(samples, key=lambda item: item["children_rss_bytes"])
        peak_total = max(samples, key=lambda item: item["total_tracked_rss_bytes"])

        child_peaks_sorted = sorted(child_peaks, key=lambda item: item["peak_rss_bytes"], reverse=True)
        summary_json = {
            "peak_main_rss_gb": _bytes_to_gb(peak_main["main_rss_bytes"]),
            "peak_main_label": peak_main["label"],
            "peak_main_timestamp": peak_main["timestamp_iso"],
            "peak_children_rss_gb": _bytes_to_gb(peak_children["children_rss_bytes"]),
            "peak_children_label": peak_children["label"],
            "peak_children_timestamp": peak_children["timestamp_iso"],
            "peak_total_tracked_rss_gb": _bytes_to_gb(peak_total["total_tracked_rss_bytes"]),
            "peak_total_label": peak_total["label"],
            "peak_total_timestamp": peak_total["timestamp_iso"],
            "top_child_processes_by_peak_rss": [
                {
                    "pid": item["pid"],
                    "name": item["name"],
                    "peak_rss_gb": _bytes_to_gb(item["peak_rss_bytes"]),
                    "cmdline": item["cmdline"],
                }
                for item in child_peaks_sorted[:20]
            ],
        }

        summary_json_path = self.output_dir / "memory_summary.json"
        summary_md_path = self.output_dir / "memory_summary.md"
        with summary_json_path.open("w", encoding="utf-8") as handle:
            json.dump(summary_json, handle, indent=2)

        lines = [
            "# Memory Profile Summary",
            "",
            f"- Peak main process RSS: `{summary_json['peak_main_rss_gb']:.3f} GB` at `{summary_json['peak_main_label']}` ({summary_json['peak_main_timestamp']})",
            f"- Peak children total RSS: `{summary_json['peak_children_rss_gb']:.3f} GB` at `{summary_json['peak_children_label']}` ({summary_json['peak_children_timestamp']})",
            f"- Peak total tracked RSS: `{summary_json['peak_total_tracked_rss_gb']:.3f} GB` at `{summary_json['peak_total_label']}` ({summary_json['peak_total_timestamp']})",
            "",
            "## Top Child Processes by Peak RSS",
            "",
            "| PID | Name | Peak RSS (GB) | Command |",
            "| --- | --- | ---: | --- |",
        ]
        for item in summary_json["top_child_processes_by_peak_rss"]:
            cmd = str(item["cmdline"]).replace("|", "\\|")
            lines.append(f"| {item['pid']} | {item['name']} | {item['peak_rss_gb']:.3f} | `{cmd}` |")

        with summary_md_path.open("w", encoding="utf-8") as handle:
            handle.write("\n".join(lines) + "\n")
