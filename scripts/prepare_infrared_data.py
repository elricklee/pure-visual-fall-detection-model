from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def to_infrared(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    infrared = np.stack([gray, gray, gray], axis=-1)
    return infrared


def process_image(input_path: Path, output_path: Path) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read image: {input_path}")
    
    infrared = to_infrared(image)
    cv2.imwrite(str(output_path), infrared)


def process_directory(input_dir: Path, output_dir: Path, recursive: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if recursive:
        files = sorted(input_dir.rglob("*"))
    else:
        files = sorted(input_dir.iterdir())
    
    for filepath in files:
        if filepath.is_file() and filepath.suffix.lower() in IMAGE_EXTS:
            rel_path = filepath.relative_to(input_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            process_image(filepath, output_path)
            print(f"Converted: {filepath} -> {output_path}")


def process_video(input_path: Path, output_path: Path) -> None:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {input_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        infrared = to_infrared(frame)
        writer.write(infrared)
        count += 1
    
    cap.release()
    writer.release()
    print(f"Converted video: {input_path} -> {output_path} ({count} frames)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare infrared data from visible light images/videos.")
    parser.add_argument("--input", required=True, help="Input image, video, or directory.")
    parser.add_argument("--output", required=True, help="Output path.")
    parser.add_argument("--recursive", action="store_true", help="Process directories recursively.")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        ext = input_path.suffix.lower()
        if ext in IMAGE_EXTS:
            process_image(input_path, output_path)
        else:
            process_video(input_path, output_path)
    elif input_path.is_dir():
        process_directory(input_path, output_path, args.recursive)
    else:
        raise ValueError(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()