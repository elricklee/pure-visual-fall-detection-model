from __future__ import annotations

import cv2
import numpy as np

from fall_detection.image_utils import ensure_3_channels, to_infrared


def letterbox(
    image: np.ndarray,
    size: int,
    padding_value: int = 114,
) -> np.ndarray:
    """Resize and pad an image to a fixed square, matching YOLO inference."""
    if size <= 0:
        raise ValueError("size must be positive")

    image = ensure_3_channels(image)
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")

    ratio = min(size / height, size / width)
    resized_width = max(1, round(width * ratio))
    resized_height = max(1, round(height * ratio))
    if (resized_width, resized_height) != (width, height):
        image = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

    horizontal_padding = size - resized_width
    vertical_padding = size - resized_height
    left = round(horizontal_padding / 2 - 0.1)
    right = round(horizontal_padding / 2 + 0.1)
    top = round(vertical_padding / 2 - 0.1)
    bottom = round(vertical_padding / 2 + 0.1)
    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(padding_value, padding_value, padding_value),
    )


def prepare_onnx_input(
    image: np.ndarray,
    size: int = 384,
    infrared: bool = False,
) -> np.ndarray:
    """Convert a BGR image into normalized RGB NCHW float32 ONNX input."""
    image = ensure_3_channels(image)
    if infrared:
        image = to_infrared(image)
    image = letterbox(image, size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = image.transpose(2, 0, 1)
    return np.ascontiguousarray(tensor[np.newaxis], dtype=np.float32) / 255.0
