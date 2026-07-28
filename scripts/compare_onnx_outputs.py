from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fall_detection.onnx_preprocess import prepare_onnx_input
from scripts.quantize_onnx import IMAGE_EXTENSIONS, evenly_select


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare FP32 and quantized ONNX raw outputs on held-out images."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/fall_pose/images/val"),
    )
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model/int8_output_comparison.json"),
    )
    return parser.parse_args()


def cosine_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    left = reference.reshape(-1).astype(np.float64)
    right = candidate.reshape(-1).astype(np.float64)
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 1.0 if np.array_equal(left, right) else 0.0
    return float(np.dot(left, right) / denominator)


def main() -> int:
    args = parse_args()
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required") from exc

    for path in (args.reference, args.candidate):
        if not path.is_file():
            raise SystemExit(f"model not found: {path}")
    if not args.images.is_dir():
        raise SystemExit(f"image directory not found: {args.images}")

    paths = sorted(
        path
        for path in args.images.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    selected = evenly_select(paths, min(args.samples, len(paths)))
    reference_session = ort.InferenceSession(
        str(args.reference), providers=["CPUExecutionProvider"]
    )
    candidate_session = ort.InferenceSession(
        str(args.candidate), providers=["CPUExecutionProvider"]
    )
    reference_input = reference_session.get_inputs()[0].name
    candidate_input = candidate_session.get_inputs()[0].name

    rows = []
    for path in selected:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"failed to read validation image: {path}")
        for infrared in (False, True):
            tensor = prepare_onnx_input(
                image,
                size=args.image_size,
                infrared=infrared,
            )
            reference = reference_session.run(None, {reference_input: tensor})[0]
            candidate = candidate_session.run(None, {candidate_input: tensor})[0]
            if reference.shape != candidate.shape:
                raise RuntimeError(
                    f"output shape mismatch: {reference.shape} vs {candidate.shape}"
                )
            difference = np.abs(reference - candidate)
            rows.append(
                {
                    "image": str(path.resolve()),
                    "mode": "pseudo_infrared" if infrared else "visible",
                    "mean_absolute_error": float(difference.mean()),
                    "root_mean_squared_error": float(
                        np.sqrt(np.mean(np.square(reference - candidate)))
                    ),
                    "max_absolute_error": float(difference.max()),
                    "cosine_similarity": cosine_similarity(reference, candidate),
                    "reference_top_person_score": float(reference[:, 4, :].max()),
                    "candidate_top_person_score": float(candidate[:, 4, :].max()),
                }
            )

    def aggregate(mode: str | None) -> dict:
        subset = rows if mode is None else [row for row in rows if row["mode"] == mode]
        return {
            "samples": len(subset),
            "mean_absolute_error": round(
                sum(row["mean_absolute_error"] for row in subset) / len(subset),
                8,
            ),
            "root_mean_squared_error": round(
                math.sqrt(
                    sum(row["root_mean_squared_error"] ** 2 for row in subset)
                    / len(subset)
                ),
                8,
            ),
            "max_absolute_error": round(
                max(row["max_absolute_error"] for row in subset),
                8,
            ),
            "mean_cosine_similarity": round(
                sum(row["cosine_similarity"] for row in subset) / len(subset),
                8,
            ),
            "mean_top_person_score_delta": round(
                sum(
                    abs(
                        row["reference_top_person_score"]
                        - row["candidate_top_person_score"]
                    )
                    for row in subset
                )
                / len(subset),
                8,
            ),
        }

    payload = {
        "reference_model": str(args.reference.resolve()),
        "candidate_model": str(args.candidate.resolve()),
        "validation_directory": str(args.images.resolve()),
        "output_shape": reference_session.get_outputs()[0].shape,
        "aggregate": {
            "all": aggregate(None),
            "visible": aggregate("visible"),
            "pseudo_infrared": aggregate("pseudo_infrared"),
        },
        "limitations": [
            "Raw tensor similarity is a quantization regression check, not pose mAP.",
            "Pseudo-infrared inputs are grayscale-replicated visible images, not real infrared sensor data.",
        ],
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
