from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SKELETON = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO pose labels.")
    parser.add_argument("--root", default="datasets/fall_pose", help="Dataset root.")
    parser.add_argument("--output", default="runs/label_review", help="Output folder.")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Optional images per split.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def iter_images(root: Path, split: str) -> list[Path]:
    image_dir = root / "images" / split
    return sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def label_path_for(root: Path, split: str, image_path: Path) -> Path:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    return (label_dir / image_path.relative_to(image_dir)).with_suffix(".txt")


def parse_label_line(line: str) -> tuple[np.ndarray, np.ndarray]:
    values = [float(v) for v in line.split()]
    if len(values) != 56:
        raise ValueError(f"expected 56 values, got {len(values)}")
    box = np.asarray(values[1:5], dtype=np.float32)
    keypoints = np.asarray(values[5:], dtype=np.float32).reshape(17, 3)
    return box, keypoints


def draw_box(image: np.ndarray, box: np.ndarray) -> None:
    h, w = image.shape[:2]
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray) -> None:
    h, w = image.shape[:2]
    pts = keypoints.copy()
    pts[:, 0] *= w
    pts[:, 1] *= h

    for a, b in SKELETON:
        if pts[a, 2] > 0 and pts[b, 2] > 0:
            cv2.line(
                image,
                (int(pts[a, 0]), int(pts[a, 1])),
                (int(pts[b, 0]), int(pts[b, 1])),
                (255, 220, 0),
                2,
            )

    for index, (x, y, visible) in enumerate(pts):
        if visible <= 0:
            continue
        color = (0, 180, 255) if visible >= 2 else (180, 180, 180)
        cv2.circle(image, (int(x), int(y)), 3, color, -1)
        cv2.putText(
            image,
            str(index),
            (int(x) + 4, int(y) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA,
        )


def visualize_image(root: Path, split: str, image_path: Path, output_root: Path) -> bool:
    label_path = label_path_for(root, split, image_path)
    image = cv2.imread(str(image_path))
    if image is None or not label_path.exists():
        return False

    lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        out = image.copy()
        cv2.putText(out, "EMPTY LABEL", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    else:
        out = image.copy()
        for line in lines:
            box, keypoints = parse_label_line(line)
            draw_box(out, box)
            draw_keypoints(out, keypoints)

    dst = output_root / split / image_path.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), out)
    return True


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()
    output_root = Path(args.output)
    if not output_root.is_absolute():
        output_root = (Path.cwd() / output_root).resolve()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    rng = random.Random(args.seed)
    total = 0

    for split in splits:
        images = iter_images(root, split)
        if args.limit > 0 and len(images) > args.limit:
            images = rng.sample(images, args.limit)
        written = sum(visualize_image(root, split, image, output_root) for image in images)
        total += written
        print(f"{split}: wrote {written} review images")

    print(f"output: {output_root}")
    print(f"total: {total}")


if __name__ == "__main__":
    main()
