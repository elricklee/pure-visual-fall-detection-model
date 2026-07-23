from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset folder skeleton.")
    parser.add_argument(
        "--root",
        default="datasets/fall_pose",
        help="Dataset root to create, relative to repo or absolute path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    if not root.is_absolute():
        root = (Path.cwd() / root).resolve()

    folders = [
        root / "images" / "train",
        root / "images" / "val",
        root / "images" / "test",
        root / "labels" / "train",
        root / "labels" / "val",
        root / "labels" / "test",
        root / "videos" / "raw",
        root / "videos" / "demo",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    print(f"created dataset skeleton at: {root}")
    for folder in folders:
        print(f"- {folder}")


if __name__ == "__main__":
    main()
