from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Iterable

import numpy as np


COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


@dataclass(frozen=True)
class PoseMetrics:
    bbox_aspect: float | None
    torso_angle_from_horizontal: float | None
    shoulder_hip_vertical_gap_ratio: float | None
    visible_keypoints: int


def as_keypoint_array(keypoints: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(keypoints, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] != 17 or arr.shape[1] not in (2, 3):
        raise ValueError("keypoints must have shape (17, 2) or (17, 3)")
    if arr.shape[1] == 2:
        arr = np.concatenate([arr, np.ones((17, 1), dtype=np.float32)], axis=1)
    return arr


def visible_point(keypoints: np.ndarray, name: str, min_conf: float) -> np.ndarray | None:
    idx = COCO_KEYPOINTS[name]
    point = keypoints[idx]
    if point[2] < min_conf:
        return None
    return point[:2]


def midpoint(points: list[np.ndarray | None]) -> np.ndarray | None:
    valid = [point for point in points if point is not None]
    if not valid:
        return None
    return np.mean(np.stack(valid, axis=0), axis=0)


def bbox_aspect_ratio(box_xyxy: Iterable[float] | None) -> float | None:
    if box_xyxy is None:
        return None
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    if height <= 1e-6:
        return None
    return width / height


def estimate_pose_metrics(
    keypoints: Iterable[Iterable[float]],
    box_xyxy: Iterable[float] | None = None,
    min_conf: float = 0.25,
) -> PoseMetrics:
    kpts = as_keypoint_array(keypoints)
    visible_count = int(np.sum(kpts[:, 2] >= min_conf))

    left_shoulder = visible_point(kpts, "left_shoulder", min_conf)
    right_shoulder = visible_point(kpts, "right_shoulder", min_conf)
    left_hip = visible_point(kpts, "left_hip", min_conf)
    right_hip = visible_point(kpts, "right_hip", min_conf)

    shoulder_center = midpoint([left_shoulder, right_shoulder])
    hip_center = midpoint([left_hip, right_hip])

    torso_angle = None
    vertical_gap_ratio = None
    aspect = bbox_aspect_ratio(box_xyxy)

    if shoulder_center is not None and hip_center is not None:
        dx = float(hip_center[0] - shoulder_center[0])
        dy = float(hip_center[1] - shoulder_center[1])
        torso_angle = abs(degrees(atan2(dy, dx)))
        if torso_angle > 90.0:
            torso_angle = 180.0 - torso_angle

        if box_xyxy is not None:
            _, y1, _, y2 = [float(v) for v in box_xyxy]
            bbox_height = max(1.0, y2 - y1)
            vertical_gap_ratio = abs(dy) / bbox_height

    return PoseMetrics(
        bbox_aspect=aspect,
        torso_angle_from_horizontal=torso_angle,
        shoulder_hip_vertical_gap_ratio=vertical_gap_ratio,
        visible_keypoints=visible_count,
    )
