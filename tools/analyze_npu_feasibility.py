#!/usr/bin/env python3
"""Create a board-independent NPU deployment feasibility report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# These operators are commonly seen in deployable CNNs. Presence in this list
# is not a compatibility guarantee; the exact target SDK must still be checked.
COMMON_EDGE_OPS = {
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Cast",
    "Clip",
    "Concat",
    "Constant",
    "Conv",
    "Div",
    "Exp",
    "Flatten",
    "Gather",
    "GlobalAveragePool",
    "LeakyRelu",
    "MatMul",
    "MaxPool",
    "Mul",
    "Pad",
    "Pow",
    "ReduceMean",
    "Relu",
    "Reshape",
    "Resize",
    "Shape",
    "Sigmoid",
    "Slice",
    "Softmax",
    "Split",
    "Squeeze",
    "Sub",
    "Transpose",
    "Unsqueeze",
}


def weight_storage_mb(parameter_count: int) -> dict:
    return {
        "fp32_mb": round(parameter_count * 4 / 1_000_000, 4),
        "fp16_mb": round(parameter_count * 2 / 1_000_000, 4),
        "int8_mb": round(parameter_count / 1_000_000, 4),
    }


def analyze(model_audit: dict, benchmark: dict | None = None) -> dict:
    parameter_count = int(model_audit["parameter_count"])
    operators = set(model_audit.get("operator_counts", {}))
    uncommon = sorted(operators - COMMON_EDGE_OPS)
    storage = weight_storage_mb(parameter_count)

    report = {
        "model_path": model_audit.get("model_path"),
        "parameter_count": parameter_count,
        "parameter_count_million": round(parameter_count / 1_000_000, 4),
        "weight_storage_estimate": storage,
        "design_checks": {
            "parameters_le_20m": parameter_count <= 20_000_000,
            "internal_target_parameters_le_10m": parameter_count <= 10_000_000,
            "fp32_weights_le_80mb": storage["fp32_mb"] <= 80.0,
            "int8_weights_le_20mb": storage["int8_mb"] <= 20.0,
        },
        "operator_review": {
            "all_operators": sorted(operators),
            "operators_needing_priority_target_sdk_review": uncommon,
            "status": "target_sdk_review_required"
            if uncommon
            else "common_ops_but_target_sdk_review_still_required",
        },
        "verification": {
            "parameter_count": "measured_from_onnx",
            "weight_storage": "calculated_from_parameter_count",
            "pc_latency": "not_provided",
            "npu_latency": "unverified_without_physical_board",
            "npu_runtime_memory": "unverified_without_physical_board",
        },
        "competition_targets": {
            "end_to_end_latency_ms": 100.0,
            "npu_memory_mb": 20.0,
        },
        "notes": [
            "INT8权重体积只计算参数存储，不包含激活、缓存和运行时开销。",
            "PC端ONNX Runtime速度不能替代真实NPU速度。",
            "算子最终是否支持，需在确定芯片后通过厂商转换工具验证。",
        ],
    }

    if benchmark:
        latency = benchmark.get("latency", {})
        report["pc_reference_benchmark"] = {
            "provider": benchmark.get("selected_providers"),
            "mean_ms": latency.get("mean_ms"),
            "p95_ms": latency.get("p95_ms"),
            "input": benchmark.get("input"),
            "scope": "pc_reference_not_npu_measurement",
        }
        report["verification"]["pc_latency"] = "measured_reference"

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="分析通用NPU部署可行性")
    parser.add_argument("model_audit", type=Path)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.model_audit.is_file():
        parser.error(f"模型审计文件不存在：{args.model_audit}")
    if args.benchmark and not args.benchmark.is_file():
        parser.error(f"测速文件不存在：{args.benchmark}")

    audit = json.loads(args.model_audit.read_text(encoding="utf-8-sig"))
    benchmark = (
        json.loads(args.benchmark.read_text(encoding="utf-8-sig"))
        if args.benchmark
        else None
    )
    report = analyze(audit, benchmark)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
