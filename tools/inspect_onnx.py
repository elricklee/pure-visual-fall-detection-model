#!/usr/bin/env python3
"""Inspect an ONNX model and emit deployment-relevant metadata."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _shape(value_info) -> list[int | str | None]:
    dims: list[int | str | None] = []
    tensor_type = value_info.type.tensor_type
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            dims.append(str(dim.dim_param))
        else:
            dims.append(None)
    return dims


def inspect_model(model_path: Path) -> dict:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as exc:
        raise SystemExit(
            "缺少 onnx。请执行：python -m pip install -r "
            "requirements-benchmark.txt"
        ) from exc

    model = onnx.load(str(model_path), load_external_data=False)
    onnx.checker.check_model(model)

    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}
    parameter_count = 0
    tensor_bytes = 0
    for tensor in initializers.values():
        array = numpy_helper.to_array(tensor)
        parameter_count += int(array.size)
        tensor_bytes += int(array.nbytes)

    ops = Counter(node.op_type for node in graph.node)
    file_size = model_path.stat().st_size

    return {
        "model_path": str(model_path.resolve()),
        "ir_version": int(model.ir_version),
        "opsets": [
            {"domain": item.domain or "ai.onnx", "version": int(item.version)}
            for item in model.opset_import
        ],
        "parameter_count": parameter_count,
        "parameter_count_million": round(parameter_count / 1_000_000, 4),
        "initializer_bytes": tensor_bytes,
        "initializer_mb": round(tensor_bytes / 1_000_000, 4),
        "file_bytes": file_size,
        "file_mb": round(file_size / 1_000_000, 4),
        "inputs": [
            {
                "name": item.name,
                "shape": _shape(item),
                "element_type": int(item.type.tensor_type.elem_type),
            }
            for item in graph.input
            if item.name not in initializers
        ],
        "outputs": [
            {
                "name": item.name,
                "shape": _shape(item),
                "element_type": int(item.type.tensor_type.elem_type),
            }
            for item in graph.output
        ],
        "node_count": len(graph.node),
        "operator_counts": dict(sorted(ops.items())),
        "hard_limit_checks": {
            "parameters_le_20m": parameter_count <= 20_000_000,
            "fp32_weight_le_80mb": tensor_bytes <= 80_000_000,
        },
        "warnings": [
            "算子是否可在 NPU 上运行，必须使用目标 SDK 转换日志确认。",
            "文件大小与运行时 NPU 内存不是同一个指标。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="审计 ONNX 模型")
    parser.add_argument("model", type=Path)
    parser.add_argument("--output", type=Path, help="JSON 输出路径")
    args = parser.parse_args()

    if not args.model.is_file():
        parser.error(f"模型不存在：{args.model}")

    report = inspect_model(args.model)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
