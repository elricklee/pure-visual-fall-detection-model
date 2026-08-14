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

    def test_suspect_cancelled_by_normal_streak(self) -> None:
        """SUSPECT_FALL should go back to NORMAL if enough normal frames pass."""
        machine = TemporalFallStateMachine(
            TemporalStateConfig(window=5, confirm_votes=5, recovery_votes=3)
        )
        box = [0, 0, 100, 200]

        machine.update(decision(0.9, True), box, 400)
        self.assertEqual(machine.state, FallState.SUSPECT_FALL)

        machine.update(decision(0.1, False), box, 400)
        machine.update(decision(0.1, False), box, 400)
        cancelled = machine.update(decision(0.1, False), box, 400)

        self.assertEqual(cancelled.state, FallState.NORMAL)

    def test_recovery_reconfirm_on_fall(self) -> None:
        """RECOVERY should go back to FALL_CONFIRMED if a fall is detected."""
        machine = TemporalFallStateMachine(
            TemporalStateConfig(
                window=3,
                confirm_votes=1,
                recovery_start_votes=1,
                recovery_votes=5,
                min_hold_frames=1,
            )
        )
        box = [0, 0, 100, 200]

        # Enter SUSPECT_FALL, then FALL_CONFIRMED
        machine.update(decision(0.9, True), box, 400)
        machine.update(decision(0.9, True), box, 400)
        self.assertEqual(machine.state, FallState.FALL_CONFIRMED)

        # Hold + enter RECOVERY
        machine.update(decision(0.1, False), box, 400)
        machine.update(decision(0.1, False), box, 400)
        self.assertEqual(machine.state, FallState.RECOVERY)

        # Reconfirm
        reconfirm = machine.update(decision(0.9, True), box, 400)
        self.assertEqual(reconfirm.state, FallState.FALL_CONFIRMED)
        self.assertTrue(reconfirm.is_fall_confirmed)

    def test_dynamic_suspect_triggered_by_fast_down(self) -> None:
        """A rapid downward center_y movement should trigger dynamic_suspect."""
        machine = TemporalFallStateMachine(
            TemporalStateConfig(
                window=5,
                confirm_votes=3,
                fast_down_delta_threshold=0.02,
            )
        )

        # First frame: person at top
        first = machine.update(decision(0.1, False), [0, 100, 100, 200], 400)
        self.assertEqual(first.state, FallState.NORMAL)

        # Second frame: person dropped significantly (center_y moves down)
        second = machine.update(decision(0.1, False), [0, 300, 100, 400], 400)
        self.assertTrue(second.dynamic_suspect)
        self.assertEqual(second.state, FallState.SUSPECT_FALL)

    def test_dynamic_suspect_triggered_by_torso_angle_drop(self) -> None:
        """A rapid torso angle drop should trigger dynamic_suspect."""
        machine = TemporalFallStateMachine(
            TemporalStateConfig(
                window=5,
                confirm_votes=3,
                torso_angle_drop_threshold=10.0,
            )
        )
        box = [0, 0, 100, 200]

        # Upright person
        machine.update(decision(0.1, False, torso_angle=80.0), box, 400)
        # Person falls (torso angle drops by > 10)
        second = machine.update(decision(0.1, False, torso_angle=60.0), box, 400)
        self.assertTrue(second.dynamic_suspect)

    def test_reset_returns_to_normal(self) -> None:
        machine = TemporalFallStateMachine(
            TemporalStateConfig(window=3, confirm_votes=1)
        )
        box = [0, 0, 100, 200]

        machine.update(decision(0.9, True), box, 400)
        machine.update(decision(0.9, True), box, 400)
        self.assertEqual(machine.state, FallState.FALL_CONFIRMED)

        machine.reset()
        self.assertEqual(machine.state, FallState.NORMAL)

    def test_normal_stays_normal_without_fall(self) -> None:
        machine = TemporalFallStateMachine(
            TemporalStateConfig(window=5, confirm_votes=3)
        )
        box = [0, 0, 100, 200]

        result = machine.update(decision(0.1, False), box, 400)
        self.assertEqual(result.state, FallState.NORMAL)
        self.assertFalse(result.is_fall_confirmed)

    def test_hold_frames_before_recovery(self) -> None:
        """FALL_CONFIRMED should hold for min_hold_frames before entering RECOVERY."""
        machine = TemporalFallStateMachine(
            TemporalStateConfig(
                window=3,
                confirm_votes=1,
                min_hold_frames=3,
                recovery_start_votes=1,
                recovery_votes=5,
            )
        )
        box = [0, 0, 100, 200]

        # Enter SUSPECT_FALL then FALL_CONFIRMED
        machine.update(decision(0.9, True), box, 400)
        machine.update(decision(0.9, True), box, 400)
        self.assertEqual(machine.state, FallState.FALL_CONFIRMED)

        # Not enough hold frames yet
        machine.update(decision(0.1, False), box, 400)
        self.assertEqual(machine.state, FallState.FALL_CONFIRMED)
        machine.update(decision(0.1, False), box, 400)
        self.assertEqual(machine.state, FallState.FALL_CONFIRMED)

        # Now enough hold frames + normal streak
        result = machine.update(decision(0.1, False), box, 400)
        self.assertEqual(result.state, FallState.RECOVERY)


if __name__ == "__main__":
    unittest.main()