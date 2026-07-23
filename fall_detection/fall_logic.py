from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

from .pose_features import PoseMetrics, estimate_pose_metrics


@dataclass(frozen=True)
class FallRuleConfig:
    min_visible_keypoints: int = 6
    bbox_aspect_threshold: float = 1.25
    torso_horizontal_angle_threshold: float = 35.0
    shoulder_hip_gap_threshold: float = 0.22
    fall_score_threshold: float = 0.70
    temporal_window: int = 5
    temporal_min_fall_votes: int = 3


@dataclass(frozen=True)
class FallDecision:
    is_fall: bool
    score: float
    reason: str
    metrics: PoseMetrics
    temporal_is_fall: bool = False


class FallDetector:
    """Rule-based fall classifier fed by YOLO pose keypoints."""

    def __init__(self, config: FallRuleConfig | None = None) -> None:
        self.config = config or FallRuleConfig()
        self._history: deque[bool] = deque(maxlen=self.config.temporal_window)

    def reset(self) -> None:
        self._history.clear()

    def classify(
        self,
        keypoints: Iterable[Iterable[float]],
        box_xyxy: Iterable[float] | None = None,
    ) -> FallDecision:
        metrics = estimate_pose_metrics(keypoints, box_xyxy)
        score = 0.0
        reasons: list[str] = []

        if metrics.visible_keypoints < self.config.min_visible_keypoints:
            decision = FallDecision(
                is_fall=False,
                score=0.0,
                reason="too_few_keypoints",
                metrics=metrics,
                temporal_is_fall=self._vote(False),
            )
            return decision

        if (
            metrics.bbox_aspect is not None
            and metrics.bbox_aspect >= self.config.bbox_aspect_threshold
        ):
            score += 0.35
            reasons.append("wide_bbox")

        if (
            metrics.torso_angle_from_horizontal is not None
            and metrics.torso_angle_from_horizontal
            <= self.config.torso_horizontal_angle_threshold
        ):
            score += 0.45
            reasons.append("horizontal_torso")

        if (
            metrics.shoulder_hip_vertical_gap_ratio is not None
            and metrics.shoulder_hip_vertical_gap_ratio
            <= self.config.shoulder_hip_gap_threshold
        ):
            score += 0.20
            reasons.append("compressed_shoulder_hip_gap")

        is_fall = score >= self.config.fall_score_threshold
        temporal_is_fall = self._vote(is_fall)
        return FallDecision(
            is_fall=is_fall,
            score=round(score, 3),
            reason="+".join(reasons) if reasons else "normal_pose",
            metrics=metrics,
            temporal_is_fall=temporal_is_fall,
        )

    def _vote(self, is_fall: bool) -> bool:
        self._history.append(is_fall)
        return sum(self._history) >= self.config.temporal_min_fall_votes
