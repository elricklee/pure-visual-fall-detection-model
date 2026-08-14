from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def adjust_brightness(image: np.ndarray, factor: float) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v * factor, 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def add_occlusion(image: np.ndarray, occlusion_ratio: float = 0.15) -> np.ndarray:
    h, w = image.shape[:2]
    result = image.copy()
    num_occlusions = random.randint(1, 3)
    
    for _ in range(num_occlusions):
        occl_w = int(w * random.uniform(0.05, 0.25))
        occl_h = int(h * random.uniform(0.05, 0.25))
        x = random.randint(0, max(0, w - occl_w))
        y = random.randint(0, max(0, h - occl_h))
        
        color = [random.randint(0, 50) for _ in range(3)]
        cv2.rectangle(result, (x, y), (x + occl_w, y + occl_h), color, thickness=-1)
    
    return result


def simulate_low_light(image: np.ndarray, darkness_factor: float = 0.4) -> np.ndarray:
    dark = adjust_brightness(image, darkness_factor)
    noise = np.random.normal(0, random.uniform(5, 15), image.shape).astype(np.int16)
    noisy = np.clip(np.int16(dark) + noise, 0, 255).astype(np.uint8)
    return noisy


def apply_random_enhancement(image: np.ndarray) -> np.ndarray:
    operations = [
        ("brightness_low", lambda img: adjust_brightness(img, random.uniform(0.3, 0.7))),
        ("brightness_high", lambda img: adjust_brightness(img, random.uniform(1.2, 1.8))),
        ("low_light", simulate_low_light),
        ("occlusion", add_occlusion),
        ("none", lambda img: img),
    ]
    
    choice = random.choices(operations, weights=[0.25, 0.15, 0.25, 0.25, 0.10])[0]
    op_name, op_func = choice
    enhanced = op_func(image)
    
    if random.random() < 0.3:
        enhanced = add_occlusion(enhanced)
    
    return enhanced, op_name


def process_image(input_path: Path, output_path: Path, num_variants: int = 3) -> None:
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Warning: Failed to read image: {input_path}")
        return
    
    base_name = input_path.stem
    ext = input_path.suffix
    
    for i in range(num_variants):
        enhanced, op_name = apply_random_enhancement(image)
        variant_name = f"{base_name}_enhance_{i}_{op_name}{ext}"
        variant_path = output_path.parent / variant_name
        cv2.imwrite(str(variant_path), enhanced)
        print(f"Enhanced: {input_path} -> {variant_path}")


def process_directory(input_dir: Path, output_dir: Path, num_variants: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for filepath in sorted(input_dir.rglob("*")):
        if filepath.is_file() and filepath.suffix.lower() in IMAGE_EXTS:
            rel_path = filepath.relative_to(input_dir)
            output_path = output_dir / rel_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            process_image(filepath, output_path, num_variants)


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply data augmentation for fall detection training.")
    parser.add_argument("--input", required=True, help="Input directory of images.")
    parser.add_argument("--output", required=True, help="Output directory for augmented images.")
    parser.add_argument("--num-variants", type=int, default=3, help="Number of variants per image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        raise ValueError(f"Input directory not found: {input_dir}")
    
    process_directory(input_dir, output_dir, args.num_variants)


if __name__ == "__main__":
    main()