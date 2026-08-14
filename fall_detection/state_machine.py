from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .fall_logic import FallDecision


class FallState(str, Enum):
    NORMAL = "NORMAL"
    SUSPECT_FALL = "SUSPECT_FALL"
    FALL_CONFIRMED = "FALL_CONFIRMED"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class TemporalStateConfig:
    window: int = 5
    confirm_votes: int = 3
    recovery_start_votes: int = 3
    recovery_votes: int = 8
    min_hold_frames: int = 5
    strong_score_threshold: float = 0.80
    fast_down_delta_threshold: float = 0.035
    torso_angle_drop_threshold: float = 18.0


@dataclass(frozen=True)
class TemporalStateDecision:
    state: FallState
    is_fall_confirmed: bool
    dynamic_suspect: bool
    fall_votes: int
    normal_votes: int
    center_y_delta: float | None
    torso_angle_delta: float | None
    reason: str


class TemporalFallStateMachine:
    """Temporal fall decision logic over consecutive pose observations."""

    def __init__(self, config: TemporalStateConfig | None = None) -> None:
        self.config = config or TemporalStateConfig()
        self.state = FallState.NORMAL
        self._fall_history: deque[bool] = deque(maxlen=self.config.window)
        self._normal_streak = 0
        self._hold_frames = 0
        self._prev_center_y: float | None = None
        self._prev_torso_angle: float | None = None

    def reset(self) -> None:
        self.state = FallState.NORMAL
        self._fall_history.clear()
        self._normal_streak = 0
        self._hold_frames = 0
        self._prev_center_y = None
        self._prev_torso_angle = None

    def update(
        self,
        decision: FallDecision,
        box_xyxy: Iterable[float] | None,
        frame_height: int | None,
    ) -> TemporalStateDecision:
        center_y = self._normalized_center_y(box_xyxy, frame_height)
        center_y_delta = (
            center_y - self._prev_center_y
            if center_y is not None and self._prev_center_y is not None
            else None
        )
        torso_angle = decision.metrics.torso_angle_from_horizontal
        torso_angle_delta = (
            self._prev_torso_angle - torso_angle
            if torso_angle is not None and self._prev_torso_angle is not None
            else None
        )

        dynamic_suspect = (
            center_y_delta is not None
            and center_y_delta >= self.config.fast_down_delta_threshold
        ) or (
            torso_angle_delta is not None
            and torso_angle_delta >= self.config.torso_angle_drop_threshold
        )

        single_frame_fall = decision.is_fall or (
            decision.score >= self.config.strong_score_threshold
        )
        self._fall_history.append(single_frame_fall)
        fall_votes = sum(self._fall_history)

        if single_frame_fall:
            self._normal_streak = 0
        else:
            self._normal_streak += 1

        reason = "normal"
        if self.state == FallState.NORMAL:
            if dynamic_suspect or single_frame_fall:
                self.state = FallState.SUSPECT_FALL
                reason = "enter_suspect"
        elif self.state == FallState.SUSPECT_FALL:
            if fall_votes >= self.config.confirm_votes:
                self.state = FallState.FALL_CONFIRMED
                self._hold_frames = 0
                reason = "confirm_by_votes"
            elif self._normal_streak >= self.config.recovery_votes:
                self.state = FallState.NORMAL
                reason = "suspect_cancelled"
            else:
                reason = "keep_suspect"
        elif self.state == FallState.FALL_CONFIRMED:
            self._hold_frames += 1
            if (
                self._hold_frames >= self.config.min_hold_frames
                and self._normal_streak >= self.config.recovery_start_votes
            ):
                self.state = FallState.RECOVERY
                reason = "enter_recovery"
            else:
                reason = "keep_confirmed"
        elif self.state == FallState.RECOVERY:
            if single_frame_fall:
                self.state = FallState.FALL_CONFIRMED
                self._hold_frames = 0
                reason = "reconfirm"
            elif self._normal_streak >= self.config.recovery_votes:
                self.state = FallState.NORMAL
                reason = "recovered"
            else:
                reason = "keep_recovery"

        self._prev_center_y = center_y
        self._prev_torso_angle = torso_angle

        return TemporalStateDecision(
            state=self.state,
            is_fall_confirmed=self.state == FallState.FALL_CONFIRMED,
            dynamic_suspect=dynamic_suspect,
            fall_votes=fall_votes,
            normal_votes=self._normal_streak,
            center_y_delta=round(center_y_delta, 4)
            if center_y_delta is not None
            else None,
            torso_angle_delta=round(torso_angle_delta, 4)
            if torso_angle_delta is not None
            else None,
            reason=reason,
        )

    @staticmethod
    def _normalized_center_y(
        box_xyxy: Iterable[float] | None,
        frame_height: int | None,
    ) -> float | None:
        if box_xyxy is None or not frame_height:
            return None
        _, y1, _, y2 = [float(v) for v in box_xyxy]
        return ((y1 + y2) / 2.0) / float(frame_height)
