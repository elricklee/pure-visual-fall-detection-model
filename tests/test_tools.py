from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


benchmark_onnx = load_module("benchmark_onnx", "tools/benchmark_onnx.py")
check_compliance = load_module("check_compliance", "tools/check_compliance.py")
analyze_npu = load_module(
    "analyze_npu_feasibility", "tools/analyze_npu_feasibility.py"
)


class LatencyStatsTests(unittest.TestCase):
    def test_latency_stats(self):
        result = benchmark_onnx.latency_stats([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(result["runs"], 4)
        self.assertEqual(result["mean_ms"], 25.0)
        self.assertEqual(result["median_ms"], 25.0)
        self.assertEqual(result["fps_from_mean"], 40.0)

    def test_percentile(self):
        self.assertEqual(benchmark_onnx.percentile([1.0, 2.0, 3.0], 0.5), 2.0)

    def test_process_rss_when_supported(self):
        value = benchmark_onnx.process_rss_bytes()
        self.assertTrue(value is None or value > 0)


class ComplianceTests(unittest.TestCase):
    def test_pass(self):
        data = {
            "visual_only": True,
            "supports_visible_light": True,
            "supports_infrared": True,
            "parameter_count": 5_000_000,
            "fp32_weight_mb": 20.0,
            "end_to_end_latency_ms": 80.0,
            "npu_memory_mb": 15.0,
            "camera_input_min_height": 1080,
        }
        report = check_compliance.evaluate(data)
        self.assertEqual(report["overall_status"], "verified_pass")
        self.assertEqual(report["summary"]["pass"], 8)

    def test_missing_and_failure(self):
        data = {
            "visual_only": True,
            "supports_visible_light": True,
            "supports_infrared": False,
            "parameter_count": 25_000_000,
        }
        report = check_compliance.evaluate(data)
        self.assertEqual(report["overall_status"], "not_ready")
        self.assertEqual(report["summary"]["fail"], 2)
        self.assertEqual(report["summary"]["missing"], 4)

    def test_design_targets_are_not_reported_as_measured_passes(self):
        data = {
            "visual_only": True,
            "supports_visible_light": True,
            "supports_infrared": True,
            "parameter_count": 5_000_000,
            "fp32_weight_mb": 20.0,
            "end_to_end_latency_ms": 100.0,
            "npu_memory_mb": 20.0,
            "camera_input_min_height": 1080,
            "evidence_type": {
                "end_to_end_latency_ms": "design_target",
                "npu_memory_mb": "design_target",
            },
        }
        report = check_compliance.evaluate(data)
        self.assertEqual(report["overall_status"], "design_ready_unverified")
        self.assertEqual(report["summary"]["design_target"], 2)


class FeasibilityTests(unittest.TestCase):
    def test_weight_storage_estimate(self):
        result = analyze_npu.weight_storage_mb(5_000_000)
        self.assertEqual(result["fp32_mb"], 20.0)
        self.assertEqual(result["fp16_mb"], 10.0)
        self.assertEqual(result["int8_mb"], 5.0)

    def test_uncommon_operator_is_flagged(self):
        audit = {
            "parameter_count": 5_000_000,
            "operator_counts": {"Conv": 10, "CustomGridSample": 1},
        }
        result = analyze_npu.analyze(audit)
        self.assertEqual(
            result["operator_review"][
                "operators_needing_priority_target_sdk_review"
            ],
            ["CustomGridSample"],
        )
        self.assertEqual(
            result["verification"]["npu_latency"],
            "unverified_without_physical_board",
        )


if __name__ == "__main__":
    unittest.main()
