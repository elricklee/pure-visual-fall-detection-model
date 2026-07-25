from __future__ import annotations

import cv2
import numpy as np


def to_infrared(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        gray = image
    elif len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    
    return np.stack([gray, gray, gray], axis=-1)


def ensure_3_channels(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return np.stack([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        return np.concatenate([image, image, image], axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return image
    elif len(image.shape) == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_keep_aspect(image: np.ndarray, target_size: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h == w == target_size:
        return image
    
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)