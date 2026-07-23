from __future__ import annotations

import argparse
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO pose dataset layout.")
    parser.add_argument("--data", default="configs/fall_pose.yaml", help="Dataset yaml path.")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_split(root: Path, split_value: str) -> tuple[Path, Path]:
    image_dir = root / split_value
    label_dir = root / split_value.replace("images", "labels", 1)
    return image_dir, label_dir


def validate_label_file(label_path: Path, expected_keypoints: int) -> list[str]:
    errors: list[str] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()
    expected_values = 5 + expected_keypoints * 3

    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != expected_values:
            errors.append(
                f"{label_path}:{line_no} expected {expected_values} values, got {len(parts)}"
            )
            continue
        try:
            values = [float(v) for v in parts]
        except ValueError:
            errors.append(f"{label_path}:{line_no} contains non-numeric value")
            continue
        coords = values[1:5]
        keypoint_values = values[5:]
        xy_values = [
            value
            for index, value in enumerate(keypoint_values)
            if index % 3 in (0, 1)
        ]
        if any(v < 0.0 or v > 1.0 for v in coords + xy_values):
            errors.append(f"{label_path}:{line_no} has normalized values outside [0, 1]")

    return errors


def main() -> None:
    args = parse_args()
    config_path = Path(args.data)
    cfg = load_config(config_path)
    root = Path(cfg["path"])
    if not root.is_absolute():
        config_relative = (config_path.parent / root).resolve()
        cwd_relative = (Path.cwd() / root).resolve()
        root = config_relative if config_relative.exists() else cwd_relative

    expected_keypoints = int(cfg.get("kpt_shape", [17, 3])[0])
    total_images = 0
    total_errors: list[str] = []

    for split_name in ("train", "val", "test"):
        split_value = cfg.get(split_name)
        if not split_value:
            continue
        image_dir, label_dir = resolve_split(root, split_value)
        if not image_dir.exists():
            total_errors.append(f"missing image directory: {image_dir}")
            continue
        if not label_dir.exists():
            total_errors.append(f"missing label directory: {label_dir}")
            continue

        images = sorted(p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
        total_images += len(images)
        missing_labels = 0

        for image_path in images:
            relative = image_path.relative_to(image_dir)
            label_path = (label_dir / relative).with_suffix(".txt")
            if not label_path.exists():
                missing_labels += 1
                total_errors.append(f"missing label for {image_path}: {label_path}")
                continue
            total_errors.extend(validate_label_file(label_path, expected_keypoints))

        print(f"{split_name}: images={len(images)} missing_labels={missing_labels}")

    if total_images == 0:
        total_errors.append("dataset has no images")

    if total_errors:
        print("\nValidation errors:")
        for error in total_errors[:50]:
            print(f"- {error}")
        if len(total_errors) > 50:
            print(f"... {len(total_errors) - 50} more")
        raise SystemExit(1)

    print("dataset validation passed")


if __name__ == "__main__":
    main()
