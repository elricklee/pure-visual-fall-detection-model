from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate YOLO pose pseudo-labels.")
    parser.add_argument("--root", default="datasets/fall_pose", help="Dataset root.")
    parser.add_argument("--model", default="yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-persons", type=int, default=1)
    return parser.parse_args()


def image_files(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(root: Path, split: str, image_path: Path) -> Path:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    return (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")


def normalize_box(box_xyxy, width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return [cx / width, cy / height, bw / width, bh / height]


def normalize_keypoints(kpts, width: int, height: int) -> list[float]:
    values: list[float] = []
    for x, y, conf in kpts:
        visible = 2 if float(conf) > 0.25 else 0
        values.extend([float(x) / width, float(y) / height, visible])
    return values


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    total_images = 0
    total_labels = 0

    for split in ("train", "val", "test"):
        images = image_files(root, split)
        if not images:
            continue

        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            label_path = label_path_for(root, split, image_path)
            label_path.parent.mkdir(parents=True, exist_ok=True)

            results = model.predict(
                source=str(image_path),
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )
            rows: list[str] = []
            for result in results:
                if result.boxes is None or result.keypoints is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                keypoints = result.keypoints.data.cpu().numpy()
                pairs = list(zip(boxes, keypoints))
                pairs.sort(
                    key=lambda pair: (pair[0][2] - pair[0][0])
                    * (pair[0][3] - pair[0][1]),
                    reverse=True,
                )
                for box, kpts in pairs[: args.max_persons]:
                    values = [0.0]
                    values.extend(normalize_box(box, width, height))
                    values.extend(normalize_keypoints(kpts, width, height))
                    rows.append(" ".join(f"{v:.6f}" for v in values))

            label_path.write_text("\n".join(rows), encoding="utf-8")
            total_images += 1
            total_labels += len(rows)

    print(f"processed images: {total_images}")
    print(f"written person labels: {total_labels}")


if __name__ == "__main__":
    main()
