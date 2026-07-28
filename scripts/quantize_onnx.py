from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fall_detection.onnx_preprocess import prepare_onnx_input


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def evenly_select(paths: list[Path], count: int) -> list[Path]:
    if count <= 0:
        return []
    if not paths:
        raise ValueError("no calibration images found")
    if count >= len(paths):
        return paths.copy()
    if count == 1:
        return [paths[len(paths) // 2]]

    last = len(paths) - 1
    return [paths[round(index * last / (count - 1))] for index in range(count)]


def build_calibration_entries(
    image_paths: list[Path],
    sample_count: int,
    infrared_ratio: float,
) -> list[tuple[Path, bool]]:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= infrared_ratio <= 1.0:
        raise ValueError("infrared_ratio must be between 0 and 1")

    infrared_count = round(sample_count * infrared_ratio)
    visible_count = sample_count - infrared_count
    visible = [(path, False) for path in evenly_select(image_paths, visible_count)]
    infrared = [(path, True) for path in evenly_select(image_paths, infrared_count)]

    entries: list[tuple[Path, bool]] = []
    while visible or infrared:
        if visible:
            entries.append(visible.pop(0))
        if infrared:
            entries.append(infrared.pop(0))
    return entries


class ImageCalibrationReader:
    def __init__(
        self,
        input_name: str,
        entries: list[tuple[Path, bool]],
        image_size: int,
    ) -> None:
        self.input_name = input_name
        self.entries = entries
        self.image_size = image_size
        self.index = 0

    def get_next(self) -> dict | None:
        if self.index >= len(self.entries):
            return None
        path, infrared = self.entries[self.index]
        self.index += 1
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read calibration image: {path}")
        return {
            self.input_name: prepare_onnx_input(
                image,
                size=self.image_size,
                infrared=infrared,
            )
        }

    def rewind(self) -> None:
        self.index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a static INT8 QDQ ONNX model from representative images."
    )
    parser.add_argument("model", type=Path, help="FP32 ONNX model.")
    parser.add_argument("output", type=Path, help="INT8 ONNX output.")
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=Path("datasets/fall_pose/images/train"),
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--infrared-ratio", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Use per-channel weights; an opset-12 source is upgraded to opset 13.",
    )
    parser.add_argument(
        "--keep-pose-head-fp32",
        action="store_true",
        help="Keep YOLO pose cv4 output-head Conv nodes in FP32.",
    )
    parser.add_argument(
        "--keep-output-head-fp32",
        action="store_true",
        help="Keep all YOLO model.22 output-head Conv nodes in FP32.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/model/int8_quantization.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not args.calibration_dir.is_dir():
        raise SystemExit(f"calibration directory not found: {args.calibration_dir}")

    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationMethod,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as exc:
        raise SystemExit(
            "onnx and onnxruntime are required; install requirements-benchmark.txt"
        ) from exc

    images = sorted(
        path
        for path in args.calibration_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    entries = build_calibration_entries(
        images,
        args.samples,
        args.infrared_ratio,
    )

    model = onnx.load(str(args.model), load_external_data=False)
    onnx.checker.check_model(model)
    input_name = model.graph.input[0].name
    reader = ImageCalibrationReader(input_name, entries, args.image_size)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_opset = next(
        item.version for item in model.opset_import if item.domain in ("", "ai.onnx")
    )
    if args.keep_output_head_fp32:
        excluded_nodes = [
            node.name
            for node in model.graph.node
            if node.op_type == "Conv" and "/model.22/" in node.name
        ]
    elif args.keep_pose_head_fp32:
        excluded_nodes = [
            node.name
            for node in model.graph.node
            if node.op_type == "Conv" and "/model.22/cv4." in node.name
        ]
    else:
        excluded_nodes = []
    quantization_input = args.model
    temporary_input: Path | None = None
    if args.per_channel and source_opset < 13:
        converted = onnx.version_converter.convert_version(model, 13)
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as stream:
            temporary_input = Path(stream.name)
        onnx.save(converted, str(temporary_input))
        quantization_input = temporary_input

    try:
        quantize_static(
            model_input=str(quantization_input),
            model_output=str(args.output),
            calibration_data_reader=reader,
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=args.per_channel,
            calibrate_method=CalibrationMethod.MinMax,
            op_types_to_quantize=["Conv"],
            nodes_to_exclude=excluded_nodes,
            extra_options={
                "ActivationSymmetric": False,
                "WeightSymmetric": True,
            },
        )
    finally:
        if temporary_input is not None:
            temporary_input.unlink(missing_ok=True)

    quantized = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(quantized)
    session = ort.InferenceSession(
        str(args.output),
        providers=["CPUExecutionProvider"],
    )
    if session.get_inputs()[0].shape != [1, 3, args.image_size, args.image_size]:
        raise RuntimeError("quantized model input shape changed unexpectedly")

    visible_count = sum(not infrared for _, infrared in entries)
    infrared_count = sum(infrared for _, infrared in entries)
    report = {
        "source_model": str(args.model.resolve()),
        "output_model": str(args.output.resolve()),
        "source_sha256": sha256(args.model),
        "output_sha256": sha256(args.output),
        "source_file_bytes": args.model.stat().st_size,
        "output_file_bytes": args.output.stat().st_size,
        "compression_ratio": round(
            args.output.stat().st_size / args.model.stat().st_size,
            6,
        ),
        "onnxruntime_version": ort.__version__,
        "quantization": {
            "method": "static",
            "format": "QDQ",
            "calibration": "MinMax",
            "activation_type": "QUInt8",
            "weight_type": "QInt8",
            "per_channel_weights": args.per_channel,
            "source_opset": source_opset,
            "output_opset": next(
                item.version
                for item in quantized.opset_import
                if item.domain in ("", "ai.onnx")
            ),
            "opset_upgrade_reason": (
                "Per-channel QDQ requires opset 13."
                if args.per_channel and source_opset < 13
                else None
            ),
            "operator_types": ["Conv"],
            "excluded_nodes": excluded_nodes,
            "mixed_precision_note": (
                "Backbone/neck Conv nodes are INT8 QDQ; all model.22 output-head "
                "Conv nodes remain FP32 to limit detection and keypoint accuracy loss."
                if args.keep_output_head_fp32
                else (
                    "Backbone/detection Conv nodes are INT8 QDQ; pose cv4 output-head "
                    "Conv nodes remain FP32 to limit keypoint accuracy loss."
                    if args.keep_pose_head_fp32
                    else None
                )
            ),
        },
        "calibration": {
            "directory": str(args.calibration_dir.resolve()),
            "samples": len(entries),
            "visible_samples": visible_count,
            "pseudo_infrared_samples": infrared_count,
            "image_size": args.image_size,
            "note": (
                "Pseudo-infrared samples are grayscale-replicated visible images; "
                "they validate preprocessing compatibility but are not real sensor data."
            ),
            "entries": [
                {
                    "path": str(path.resolve()),
                    "mode": "pseudo_infrared" if infrared else "visible",
                }
                for path, infrared in entries
            ],
        },
        "verification": {
            "onnx_checker": "pass",
            "cpu_execution_provider_load": "pass",
            "input_name": session.get_inputs()[0].name,
            "input_shape": session.get_inputs()[0].shape,
            "output_shape": session.get_outputs()[0].shape,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
