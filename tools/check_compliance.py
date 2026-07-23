#!/usr/bin/env python3
"""Validate measured model metrics against competition hard limits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


LIMITS = {
    "parameter_count": {
        "operator": "<=",
        "limit": 20_000_000,
        "label": "模型总参数不超过 20M",
    },
    "fp32_weight_mb": {
        "operator": "<=",
        "limit": 80.0,
        "label": "FP32 权重不超过 80MB",
    },
    "end_to_end_latency_ms": {
        "operator": "<=",
        "limit": 100.0,
        "label": "端侧端到端推理不超过 100ms",
    },
    "npu_memory_mb": {
        "operator": "<=",
        "limit": 20.0,
        "label": "推理时 NPU 存储占用不超过 20MB",
    },
    "camera_input_min_height": {
        "operator": ">=",
        "limit": 1080,
        "label": "相机图像输入分辨率达到 1080P 以上",
    },
}

BOOLEAN_REQUIREMENTS = {
    "visual_only": "纯视觉方案",
    "supports_visible_light": "支持可见光",
    "supports_infrared": "支持红外图像",
}


def evaluate(data: dict) -> dict:
    checks = []
    evidence_types = data.get("evidence_type", {})

    for key, label in BOOLEAN_REQUIREMENTS.items():
        value = data.get(key)
        evidence_type = evidence_types.get(key, "measured")
        checks.append(
            {
                "key": key,
                "label": label,
                "value": value,
                "limit": True,
                "evidence_type": evidence_type,
                "status": "pass"
                if value is True
                else ("missing" if value is None else "fail"),
            }
        )

    for key, rule in LIMITS.items():
        value = data.get(key)
        evidence_type = evidence_types.get(key, "measured")
        if value is None:
            status = "missing"
        elif rule["operator"] == "<=":
            status = "pass" if value <= rule["limit"] else "fail"
        else:
            status = "pass" if value >= rule["limit"] else "fail"
        if status == "pass" and evidence_type == "design_target":
            status = "design_target"
        checks.append(
            {
                "key": key,
                "label": rule["label"],
                "value": value,
                "operator": rule["operator"],
                "limit": rule["limit"],
                "evidence_type": evidence_type,
                "status": status,
            }
        )

    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in ("pass", "design_target", "fail", "missing")
    }
    if counts["fail"] > 0 or counts["missing"] > 0:
        overall_status = "not_ready"
    elif counts["design_target"] > 0:
        overall_status = "design_ready_unverified"
    else:
        overall_status = "verified_pass"

    return {
        "model_name": data.get("model_name"),
        "model_version": data.get("model_version"),
        "target_device": data.get("target_device"),
        "deployment_scope": data.get("deployment_scope"),
        "overall_status": overall_status,
        "summary": counts,
        "checks": checks,
        "evidence": data.get("evidence", {}),
        "disclaimer": (
            "design_target表示方案目标，不代表在真实NPU上完成了实测。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="检查赛题硬性指标")
    parser.add_argument("metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not args.metrics.is_file():
        parser.error(f"指标文件不存在：{args.metrics}")

    data = json.loads(args.metrics.read_text(encoding="utf-8-sig"))
    report = evaluate(data)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8-sig")

    if report["overall_status"] == "verified_pass":
        return 0
    if report["overall_status"] == "design_ready_unverified":
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
