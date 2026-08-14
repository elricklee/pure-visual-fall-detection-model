from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import cv2
import numpy as np

from .fall_logic import FallDecision

if TYPE_CHECKING:
    from .tracker_manager import PersonResult


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

# Color palette for per-person visualization (BGR).
PERSON_COLORS = [
    (0, 180, 0),      # green
    (0, 165, 255),    # orange
    (255, 0, 0),      # blue
    (0, 255, 255),    # yellow
    (255, 0, 255),    # magenta
    (255, 255, 0),    # cyan
    (128, 0, 128),    # purple
    (0, 128, 255),    # light orange
]


def _person_color(track_id: int) -> tuple[int, int, int]:
    return PERSON_COLORS[track_id % len(PERSON_COLORS)]


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
    text = f"{label} det={det_conf:.2f}"
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
    text = f"STATE: {state}  score={score:.2f}"
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


# ---------------------------------------------------------------------------
# Multi-person visualization
# ---------------------------------------------------------------------------


def draw_multi_decision(
    image: np.ndarray,
    box_xyxy: Iterable[float],
    decision: FallDecision,
    track_id: int,
    det_conf: float = 0.0,
) -> None:
    """Draw bbox and label for one person in multi-person mode.

    Fall detections always use red; normal uses a per-person color.
    Label includes the track_id for identification.
    """
    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    if decision.temporal_is_fall:
        status_color = (0, 0, 255)
        label = "FALL"
    elif decision.is_fall:
        status_color = (0, 165, 255)
        label = "FALL_POSE"
    else:
        status_color = _person_color(track_id)
        label = "Normal"
    cv2.rectangle(image, (x1, y1), (x2, y2), status_color, 2)
    text = f"#{track_id} {label} det={det_conf:.2f}"
    cv2.putText(
        image,
        text,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        status_color,
        2,
        cv2.LINE_AA,
    )


def draw_multi_status_banner(
    image: np.ndarray,
    person_results: list[PersonResult],
    max_display: int = 8,
) -> None:
    """Draw a multi-row status banner showing per-person state.

    Each person gets one row. Falls are highlighted in red.
    At most *max_display* rows are shown; extras are omitted.
    """
    from .tracker_manager import PersonResult  # noqa: F811 (runtime import)

    displayed = person_results[:max_display]
    n_rows = max(1, len(displayed))
    banner_height = 30 * n_rows + 10
    cv2.rectangle(image, (10, 10), (420, 10 + banner_height), (0, 0, 0), -1)

    y_offset = 35
    if not displayed:
        cv2.putText(
            image,
            "NO_PERSON",
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (128, 128, 128),
            2,
            cv2.LINE_AA,
        )
        return

    for pr in displayed:
        state_name = pr.state_decision.state.value if pr.state_decision else "VOTE"
        is_alert = pr.decision.temporal_is_fall
        color = (0, 0, 255) if is_alert else _person_color(pr.track_id)
        text = f"#{pr.track_id} {state_name} score={pr.decision.score:.2f}"
        cv2.putText(
            image,
            text,
            (20, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        y_offset += 30
