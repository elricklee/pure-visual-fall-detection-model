from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one pose ONNX model and save Ultralytics metrics."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--data", type=Path, default=Path("configs/fall_pose.yaml"))
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/int8_pose_eval")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if not args.data.is_file():
        raise SystemExit(f"dataset config not found: {args.data}")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required") from exc

    model = YOLO(str(args.model), task="pose")
    metrics = model.val(
        data=str(args.data),
        imgsz=args.image_size,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=True,
        plots=False,
        save_json=False,
        verbose=False,
    )
    payload = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "task": "pose",
        "image_size": args.image_size,
        "batch": args.batch,
        "device": args.device,
        "metrics": {
            key: float(value)
            for key, value in metrics.results_dict.items()
        },
        "speed_ms_per_image": {
            key: round(float(value), 6)
            for key, value in metrics.speed.items()
        },
        "limitations": [
            "The local validation split contains only 11 images.",
            "This result is a small-sample regression check, not final competition accuracy.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
