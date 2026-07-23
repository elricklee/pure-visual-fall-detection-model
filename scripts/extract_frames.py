from __future__ import annotations

import argparse
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos.")
    parser.add_argument("--source", required=True, help="Video file or folder.")
    parser.add_argument("--output", required=True, help="Output image folder.")
    parser.add_argument(
        "--every-n",
        type=int,
        default=5,
        help="Save one frame every N frames.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional cap on saved frames per video. 0 means no cap.",
    )
    parser.add_argument(
        "--prefix",
        default="frame",
        help="Output filename prefix.",
    )
    return parser.parse_args()


def iter_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def extract_from_video(
    video_path: Path,
    output_dir: Path,
    every_n: int,
    max_frames: int,
    prefix: str,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"skip unreadable video: {video_path}")
        return 0

    saved = 0
    frame_idx = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % every_n == 0:
            out_name = f"{prefix}_{stem}_{frame_idx:06d}.jpg"
            out_path = output_dir / out_name
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames > 0 and saved >= max_frames:
                break
        frame_idx += 1

    cap.release()
    print(f"{video_path.name}: saved {saved} frames")
    return saved


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output_dir = Path(args.output)
    videos = iter_videos(source)

    if not videos:
        raise SystemExit(f"no video files found in {source}")

    total = 0
    for video_path in videos:
        total += extract_from_video(
            video_path=video_path,
            output_dir=output_dir,
            every_n=max(1, args.every_n),
            max_frames=args.max_frames,
            prefix=args.prefix,
        )
    print(f"total saved frames: {total}")


if __name__ == "__main__":
    main()
