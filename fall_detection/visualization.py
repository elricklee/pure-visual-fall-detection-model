from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from .fall_logic import FallDecision


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


def draw_pose(
    image: np.ndarray,
    keypoints: Iterable[Iterable[float]],
    min_conf: float = 0.25,
) -> None:
    kpts = np.asarray(keypoints, dtype=np.float32)
    if kpts.shape[1] == 2:
        conf = np.ones((kpts.shape[0], 1), dtype=np.float32)
        kpts = np.concatenate([kpts, conf], axis=1)

    for a, b in SKELETON:
        if kpts[a, 2] >= min_conf and kpts[b, 2] >= min_conf:
            pt_a = (int(kpts[a, 0]), int(kpts[a, 1]))
            pt_b = (int(kpts[b, 0]), int(kpts[b, 1]))
            cv2.line(image, pt_a, pt_b, (255, 220, 0), 2)

    for x, y, conf in kpts:
        if conf >= min_conf:
            cv2.circle(image, (int(x), int(y)), 3, (0, 180, 255), -1)


def draw_decision(
    image: np.ndarray,
    box_xyxy: Iterable[float],
    decision: FallDecision,
    det_conf: float = 0.0,
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if decision.temporal_is_fall:
        color = (0, 0, 255)
        label = "FALL"
    elif decision.is_fall:
        color = (0, 165, 255)
        label = "FALL_POSE"
    else:
        color = (0, 180, 0)
        label = "NORMAL"
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    text = f"{label} conf={det_conf:.2f}"
    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status_banner(image: np.ndarray, state: str, is_alert: bool, score: float = 0.0) -> None:
    color = (0, 0, 255) if is_alert else (0, 160, 0)
    text = f"STATE: {state}  conf={score:.2f}"
    cv2.rectangle(image, (10, 10), (380, 44), (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )
