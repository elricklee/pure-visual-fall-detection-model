from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO image dataset.")
    parser.add_argument("--root", default="datasets/fall_pose", help="Dataset root.")
    parser.add_argument("--source-split", default="train", help="Split to split from.")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy instead of move.")
    return parser.parse_args()


def matching_label(root: Path, split: str, image_path: Path) -> Path:
    image_root = root / "images" / split
    label_root = root / "labels" / split
    return (label_root / image_path.relative_to(image_root)).with_suffix(".txt")


def transfer(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if copy:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    source_dir = root / "images" / args.source_split
    images = sorted(p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"no images found in {source_dir}")

    random.Random(args.seed).shuffle(images)
    test_count = int(len(images) * args.test_ratio)
    val_count = int(len(images) * args.val_ratio)

    test_images = set(images[:test_count])
    val_images = set(images[test_count : test_count + val_count])

    moved = {"val": 0, "test": 0}
    for image_path in images:
        target_split = None
        if image_path in test_images:
            target_split = "test"
        elif image_path in val_images:
            target_split = "val"

        if target_split is None:
            continue

        src_label = matching_label(root, args.source_split, image_path)
        dst_image = root / "images" / target_split / image_path.name
        dst_label = root / "labels" / target_split / src_label.name

        transfer(image_path, dst_image, args.copy)
        if src_label.exists():
            transfer(src_label, dst_label, args.copy)
        moved[target_split] += 1

    train_count = len(list((root / "images" / "train").glob("*")))
    print(f"train images: {train_count}")
    print(f"val images moved: {moved['val']}")
    print(f"test images moved: {moved['test']}")


if __name__ == "__main__":
    main()
