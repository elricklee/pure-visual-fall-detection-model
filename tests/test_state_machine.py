import unittest

from fall_detection import (
    FallDecision,
    FallState,
    TemporalFallStateMachine,
    TemporalStateConfig,
)
from fall_detection.pose_features import PoseMetrics


def decision(score: float, is_fall: bool, torso_angle: float = 80.0) -> FallDecision:
    return FallDecision(
        is_fall=is_fall,
        score=score,
        reason="test",
        metrics=PoseMetrics(
            bbox_aspect=1.0,
            torso_angle_from_horizontal=torso_angle,
            shoulder_hip_vertical_gap_ratio=0.5,
            visible_keypoints=17,
        ),
    )


class TemporalFallStateMachineTest(unittest.TestCase):
    def test_fall_requires_multiple_votes(self) -> None:
        machine = TemporalFallStateMachine(
            TemporalStateConfig(window=5, confirm_votes=3)
        )

        first = machine.update(decision(0.9, True), [0, 0, 100, 200], 400)
        second = machine.update(decision(0.9, True), [0, 20, 100, 220], 400)
        third = machine.update(decision(0.9, True), [0, 40, 100, 240], 400)

        self.assertEqual(first.state, FallState.SUSPECT_FALL)
        self.assertEqual(second.state, FallState.SUSPECT_FALL)
        self.assertEqual(third.state, FallState.FALL_CONFIRMED)
        self.assertTrue(third.is_fall_confirmed)

    def test_recovery_persists_until_normal_votes_complete(self) -> None:
        machine = TemporalFallStateMachine(
            TemporalStateConfig(
                window=3,
                confirm_votes=1,
                recovery_start_votes=2,
                recovery_votes=4,
                min_hold_frames=1,
            )
        )
        box = [0, 0, 100, 200]

        machine.update(decision(0.9, True), box, 400)
        machine.update(decision(0.9, True), box, 400)
        machine.update(decision(0.1, False), box, 400)
        recovery = machine.update(decision(0.1, False), box, 400)
        still_recovering = machine.update(decision(0.1, False), box, 400)
        normal = machine.update(decision(0.1, False), box, 400)

        self.assertEqual(recovery.state, FallState.RECOVERY)
        self.assertEqual(still_recovering.state, FallState.RECOVERY)
        self.assertEqual(normal.state, FallState.NORMAL)


if __name__ == "__main__":
    unittest.main()
