"""Multi-person tracking manager for per-person fall state isolation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from .fall_logic import FallDecision, FallDetector, FallRuleConfig
from .state_machine import TemporalFallStateMachine, TemporalStateConfig, TemporalStateDecision


@dataclass(frozen=True)
class PersonResult:
    """Complete detection result for one person in a single frame."""

    track_id: int
    decision: FallDecision
    state_decision: TemporalStateDecision | None
    box_xyxy: Iterable[float]
    keypoints: Iterable[Iterable[float]]
    det_conf: float


class TrackerManager:
    """Maintain independent FallDetector + TemporalFallStateMachine per track_id.

    Each person tracked across frames gets their own detector and state
    machine, so fall decisions are fully isolated between individuals.
    """

    def __init__(
        self,
        fall_config: FallRuleConfig | None = None,
        state_config: TemporalStateConfig | None = None,
        stale_threshold: int = 30,
        use_state_machine: bool = True,
    ) -> None:
        self._fall_config = fall_config or FallRuleConfig()
        self._state_config = state_config or TemporalStateConfig()
        self._use_state_machine = use_state_machine
        self._stale_threshold = stale_threshold
        self._trackers: dict[int, tuple[FallDetector, TemporalFallStateMachine]] = {}
        self._last_seen: dict[int, int] = {}
        self._frame_count: int = 0
        self._manual_frame_mode = False

    def set_frame(self, frame_index: int) -> None:
        """Set the current frame index for frame-based stale cleanup.

        When this is used, update() no longer advances the internal counter
        per detection. This keeps stale tracking aligned with actual video
        frames in multi-person loops.
        """
        self._frame_count = int(frame_index)
        self._manual_frame_mode = True

    @property
    def active_ids(self) -> list[int]:
        """Return all currently tracked person IDs."""
        return list(self._trackers.keys())

    def _get_or_create(
        self, track_id: int
    ) -> tuple[FallDetector, TemporalFallStateMachine]:
        if track_id not in self._trackers:
            self._trackers[track_id] = (
                FallDetector(self._fall_config),
                TemporalFallStateMachine(self._state_config),
            )
        return self._trackers[track_id]

    def update(
        self,
        track_id: int,
        keypoints: Iterable[Iterable[float]],
        box_xyxy: Iterable[float],
        frame_height: int,
        det_conf: float = 0.0,
    ) -> PersonResult:
        """Process a single person detection in the current frame.

        Creates a new detector + state_machine for previously unseen track_ids.
        Returns a PersonResult with the fall decision and optional state decision.
        """
        if not self._manual_frame_mode:
            self._frame_count += 1
        self._last_seen[track_id] = self._frame_count

        detector, state_machine = self._get_or_create(track_id)
        decision = detector.classify(keypoints, box_xyxy)

        state_decision: TemporalStateDecision | None = None
        if self._use_state_machine:
            state_decision = state_machine.update(decision, box_xyxy, frame_height)
            decision = replace(
                decision,
                temporal_is_fall=state_decision.is_fall_confirmed,
            )

        return PersonResult(
            track_id=track_id,
            decision=decision,
            state_decision=state_decision,
            box_xyxy=box_xyxy,
            keypoints=keypoints,
            det_conf=det_conf,
        )

    def cleanup_stale(self) -> list[int]:
        """Remove trackers for persons not seen for stale_threshold frames.

        Returns the list of removed track_ids.
        """
        stale_ids = [
            tid
            for tid, last in self._last_seen.items()
            if self._frame_count - last > self._stale_threshold
        ]
        for tid in stale_ids:
            del self._trackers[tid]
            del self._last_seen[tid]
        return stale_ids

    def reset_person(self, track_id: int) -> None:
        """Reset the detector and state machine for a specific person."""
        if track_id in self._trackers:
            detector, state_machine = self._trackers[track_id]
            detector.reset()
            state_machine.reset()

    def reset_all(self) -> None:
        """Reset all tracked persons' detectors and state machines."""
        for detector, state_machine in self._trackers.values():
            detector.reset()
            state_machine.reset()
