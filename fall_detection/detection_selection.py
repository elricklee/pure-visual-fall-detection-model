from __future__ import annotations

from typing import Iterable

import numpy as np


def select_primary_pose(
    boxes: Iterable[Iterable[float]],
    keypoints: Iterable[Iterable[Iterable[float]]],
    frame_width: int,
    frame_height: int,
    roi: str = "auto",
) -> tuple[np.ndarray, np.ndarray] | None:
    box_array = np.asarray(list(boxes), dtype=np.float32)
    keypoint_array = np.asarray(list(keypoints), dtype=np.float32)
    if len(box_array) == 0 or len(keypoint_array) == 0:
        return None

    selected_roi = _resolve_roi(frame_width, frame_height, roi)
    candidates = []
    for box, kpts in zip(box_array, keypoint_array):
        if not _box_in_roi(box, frame_width, selected_roi):
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        visible_score = float(np.sum(kpts[:, 2] > 0.25))
        candidates.append((area, visible_score, box, kpts))

    if not candidates and selected_roi != "full":
        return select_primary_pose(box_array, keypoint_array, frame_width, frame_height, "full")
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, box, kpts = candidates[0]
    return box, kpts


def _resolve_roi(frame_width: int, frame_height: int, roi: str) -> str:
    if roi != "auto":
        return roi
    aspect = frame_width / max(1, frame_height)
    return "right" if aspect >= 1.8 else "full"


def _box_in_roi(box: np.ndarray, frame_width: int, roi: str) -> bool:
    if roi == "full":
        return True
    x1, _, x2, _ = [float(v) for v in box]
    center_x = (x1 + x2) / 2.0
    midpoint = frame_width / 2.0
    if roi == "left":
        return center_x < midpoint
    if roi == "right":
        return center_x >= midpoint
    raise ValueError(f"unsupported roi: {roi}")
