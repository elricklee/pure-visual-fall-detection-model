from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8n-pose fall baseline.")
    parser.add_argument("--data", default="configs/fall_pose.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--model", default="artifacts/pytorch/yolov8n-pose.pt", help="Base pose model.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cuda device id, cpu, or None for auto.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="fall_pose_yolov8n")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset yaml not found: {data_path}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        plots=True,
        save=True,
    )


if __name__ == "__main__":
    main()
