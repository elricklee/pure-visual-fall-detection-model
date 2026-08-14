#!/usr/bin/env python3
"""Check deployment and benchmarking dependencies without importing them."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import sys
from pathlib import Path


PYTHON_MODULES = (
    "numpy",
    "onnx",
    "onnxruntime",
    "psutil",
    "cv2",
    "torch",
    "ultralytics",
    "rknn",
    "rknnlite",
)

COMMANDS = (
    "atc",
    "msame",
    "npu-smi",
    "rknn-toolkit2",
    "nvidia-smi",
)


def build_report() -> dict:
    modules = {
        name: bool(importlib.util.find_spec(name)) for name in PYTHON_MODULES
    }
    commands = {name: shutil.which(name) for name in COMMANDS}
    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "python_modules": modules,
        "commands": commands,
        "benchmark_ready": all(
            modules[name] for name in ("numpy", "onnx", "onnxruntime")
        ),
        "rknn_toolkit_detected": modules["rknn"] or modules["rknnlite"],
        "huawei_cann_detected": commands["atc"] is not None,
        "notes": [
            "benchmark_ready 仅代表可运行 ONNX Runtime 本机测速。",
            "RKNN 与海思 CANN 的版本必须匹配具体目标芯片和厂商 SDK。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检查部署与测速环境")
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径")
    args = parser.parse_args()

    report = build_report()
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8-sig")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
