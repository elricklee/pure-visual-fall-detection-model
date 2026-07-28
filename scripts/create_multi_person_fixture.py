from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a side-by-side multi-person functional-test video from two "
            "existing videos. The output is for pipeline validation, not accuracy claims."
        )
    )
    parser.add_argument("--left", required=True, type=Path)
    parser.add_argument("--right", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tile-width", type=int, default=640)
    parser.add_argument("--tile-height", type=int, default=480)
    parser.add_argument("--left-roi", choices=["full", "left", "right"], default="full")
    parser.add_argument("--right-roi", choices=["full", "left", "right"], default="full")
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def _open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open input video: {path}")
    return cap


def _apply_roi(frame: np.ndarray, roi: str) -> np.ndarray:
    if roi == "full":
        return frame
    width = frame.shape[1]
    midpoint = width // 2
    if roi == "left":
        return frame[:, :midpoint]
    return frame[:, midpoint:]


def main() -> None:
    args = parse_args()
    if args.tile_width <= 0 or args.tile_height <= 0:
        raise SystemExit("tile width and height must be positive")

    left_cap = _open_video(args.left)
    right_cap = _open_video(args.right)
    left_fps = float(left_cap.get(cv2.CAP_PROP_FPS))
    right_fps = float(right_cap.get(cv2.CAP_PROP_FPS))
    valid_fps = [fps for fps in (left_fps, right_fps) if fps > 0]
    fps = min(valid_fps) if valid_fps else 25.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_size = (args.tile_width * 2, args.tile_height)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        output_size,
    )
    if not writer.isOpened():
        left_cap.release()
        right_cap.release()
        raise SystemExit(f"Failed to open output video writer: {args.output}")

    frames = 0
    while True:
        left_ok, left_frame = left_cap.read()
        right_ok, right_frame = right_cap.read()
        if not left_ok or not right_ok:
            break
        if args.max_frames > 0 and frames >= args.max_frames:
            break

        left_frame = _apply_roi(left_frame, args.left_roi)
        right_frame = _apply_roi(right_frame, args.right_roi)
        left_frame = cv2.resize(left_frame, (args.tile_width, args.tile_height))
        right_frame = cv2.resize(right_frame, (args.tile_width, args.tile_height))
        composite = np.hstack((left_frame, right_frame))
        cv2.putText(
            composite,
            f"LEFT: {args.left.stem}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            composite,
            f"RIGHT: {args.right.stem}",
            (args.tile_width + 16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(composite)
        frames += 1

    left_cap.release()
    right_cap.release()
    writer.release()

    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": frames,
                "fps": fps,
                "width": output_size[0],
                "height": output_size[1],
                "left_roi": args.left_roi,
                "right_roi": args.right_roi,
                "purpose": "multi-person functional pipeline test only",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
