#!/usr/bin/env python3
"""Benchmark an ONNX model with reproducible warm-up and latency statistics."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.wintypes
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Iterable


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def latency_stats(values_ms: Iterable[float]) -> dict:
    values = list(values_ms)
    if not values:
        raise ValueError("at least one latency value is required")
    mean_ms = statistics.fmean(values)
    return {
        "runs": len(values),
        "min_ms": round(min(values), 4),
        "mean_ms": round(mean_ms, 4),
        "median_ms": round(statistics.median(values), 4),
        "p90_ms": round(percentile(values, 0.90), 4),
        "p95_ms": round(percentile(values, 0.95), 4),
        "p99_ms": round(percentile(values, 0.99), 4),
        "max_ms": round(max(values), 4),
        "std_ms": round(statistics.pstdev(values), 4),
        "fps_from_mean": round(1000.0 / mean_ms, 3) if mean_ms > 0 else None,
    }


def process_rss_bytes() -> int | None:
    """Return resident memory without requiring psutil."""
    if os.name == "nt":
        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.wintypes.DWORD),
                ("PageFaultCount", ctypes.wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.wintypes.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.WorkingSetSize) if ok else None

    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def _resolve_shape(shape, fallback: int) -> list[int]:
    resolved = []
    for index, dim in enumerate(shape):
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        elif index == 0:
            resolved.append(1)
        elif len(shape) == 4 and index in (2, 3):
            resolved.append(fallback)
        else:
            resolved.append(3 if len(shape) == 4 and index == 1 else 1)
    return resolved


def benchmark(
    model_path: Path,
    warmup: int,
    runs: int,
    image_size: int,
    input_npy: Path | None,
    providers: list[str] | None,
) -> dict:
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit(
            "缺少 numpy 或 onnxruntime。请安装 requirements-benchmark.txt"
        ) from exc

    available = ort.get_available_providers()
    selected = providers or [
        name
        for name in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if name in available
    ]
    if not selected:
        raise SystemExit(f"没有可用执行提供器。检测到：{available}")

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=selected
    )

    inputs = session.get_inputs()
    if len(inputs) != 1:
        raise SystemExit(
            f"当前工具要求单输入模型，实际输入数量为 {len(inputs)}。"
        )
    model_input = inputs[0]

    if input_npy:
        tensor = np.load(input_npy)
    else:
        shape = _resolve_shape(model_input.shape, image_size)
        if model_input.type == "tensor(uint8)":
            tensor = np.random.default_rng(2026).integers(
                0, 256, size=shape, dtype=np.uint8
            )
        else:
            tensor = np.random.default_rng(2026).random(
                shape, dtype=np.float32
            )

    feed = {model_input.name: tensor}
    output_names = [item.name for item in session.get_outputs()]

    rss_before = process_rss_bytes()
    for _ in range(warmup):
        session.run(output_names, feed)

    latencies = []
    peak_rss = process_rss_bytes()
    for _ in range(runs):
        started = time.perf_counter_ns()
        session.run(output_names, feed)
        ended = time.perf_counter_ns()
        latencies.append((ended - started) / 1_000_000)
        current_rss = process_rss_bytes()
        if current_rss is not None:
            peak_rss = max(peak_rss or 0, current_rss)

    rss_after = process_rss_bytes()
    return {
        "model_path": str(model_path.resolve()),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "onnxruntime_version": ort.__version__,
        "available_providers": available,
        "selected_providers": session.get_providers(),
        "input": {
            "name": model_input.name,
            "declared_shape": model_input.shape,
            "actual_shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "source": str(input_npy.resolve()) if input_npy else "seeded_random",
        },
        "warmup_runs": warmup,
        "latency": latency_stats(latencies),
        "process_memory": {
            "rss_before_mb": round(rss_before / 1_000_000, 4)
            if rss_before is not None
            else None,
            "rss_after_mb": round(rss_after / 1_000_000, 4)
            if rss_after is not None
            else None,
            "peak_observed_rss_mb": round(peak_rss / 1_000_000, 4)
            if peak_rss is not None
            else None,
            "metric_scope": "host_process_rss_not_npu_memory",
        },
        "notes": [
            "测速不包含视频解码、图像缩放、NMS、时序判别和报警输出。",
            "正式赛题数据应补充端到端耗时，并在目标 NPU 上重新测量。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="ONNX Runtime 推理测速")
    parser.add_argument("model", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--input-npy", type=Path)
    parser.add_argument(
        "--providers",
        nargs="+",
        help="指定 ONNX Runtime providers，默认优先 CUDA 后 CPU",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"模型不存在：{args.model}")
    if args.warmup < 0 or args.runs <= 0:
        parser.error("warmup 必须不小于 0，runs 必须大于 0")
    if args.input_npy and not args.input_npy.is_file():
        parser.error(f"输入文件不存在：{args.input_npy}")

    report = benchmark(
        args.model,
        args.warmup,
        args.runs,
        args.image_size,
        args.input_npy,
        args.providers,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
