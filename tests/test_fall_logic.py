import unittest

import numpy as np

from fall_detection import FallDetector, FallRuleConfig
from fall_detection.fall_logic import ramp_down, ramp_up


def empty_pose() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


class RampUpTest(unittest.TestCase):
    def test_below_start_is_zero(self) -> None:
        self.assertEqual(ramp_up(1.0, 1.25, 1.80), 0.0)

    def test_at_start_is_zero(self) -> None:
        self.assertEqual(ramp_up(1.25, 1.25, 1.80), 0.0)

    def test_at_full_is_one(self) -> None:
        self.assertEqual(ramp_up(1.80, 1.25, 1.80), 1.0)

    def test_above_full_is_one(self) -> None:
        self.assertEqual(ramp_up(2.0, 1.25, 1.80), 1.0)

    def test_midpoint_is_half(self) -> None:
        self.assertAlmostEqual(ramp_up(1.525, 1.25, 1.80), 0.5, places=5)


class RampDownTest(unittest.TestCase):
    def test_below_full_is_one(self) -> None:
        self.assertEqual(ramp_down(20.0, 35.0, 75.0), 1.0)

    def test_at_full_is_one(self) -> None:
        self.assertEqual(ramp_down(35.0, 35.0, 75.0), 1.0)

    def test_at_end_is_zero(self) -> None:
        self.assertEqual(ramp_down(75.0, 35.0, 75.0), 0.0)

    def test_above_end_is_zero(self) -> None:
        self.assertEqual(ramp_down(90.0, 35.0, 75.0), 0.0)

    def test_midpoint_is_half(self) -> None:
        self.assertAlmostEqual(ramp_down(55.0, 35.0, 75.0), 0.5, places=5)


class FallLogicTest(unittest.TestCase):
    def test_score_ramps_are_continuous(self) -> None:
        self.assertAlmostEqual(ramp_up(1.525, 1.25, 1.80), 0.5)
        self.assertAlmostEqual(ramp_down(55.0, 35.0, 75.0), 0.5)
        self.assertEqual(ramp_up(1.25, 1.25, 1.80), 0.0)
        self.assertEqual(ramp_down(35.0, 35.0, 75.0), 1.0)

    def test_horizontal_torso_and_wide_box_is_fall(self) -> None:
        pose = empty_pose()
        pose[5] = [120, 100, 0.9]
        pose[6] = [120, 140, 0.9]
        pose[11] = [260, 105, 0.9]
        pose[12] = [260, 145, 0.9]
        pose[15] = [320, 120, 0.9]
        pose[16] = [320, 160, 0.9]

        detector = FallDetector(FallRuleConfig(temporal_window=1, temporal_min_fall_votes=1))
        decision = detector.classify(pose, [90, 80, 360, 180])

        self.assertTrue(decision.is_fall)
        self.assertTrue(decision.temporal_is_fall)

    def test_vertical_torso_is_normal(self) -> None:
        pose = empty_pose()
        pose[5] = [140, 80, 0.9]
        pose[6] = [180, 80, 0.9]
        pose[11] = [145, 220, 0.9]
        pose[12] = [185, 220, 0.9]
        pose[15] = [145, 360, 0.9]
        pose[16] = [185, 360, 0.9]

        detector = FallDetector(FallRuleConfig(temporal_window=1, temporal_min_fall_votes=1))
        decision = detector.classify(pose, [100, 50, 220, 390])

        self.assertFalse(decision.is_fall)
        self.assertFalse(decision.temporal_is_fall)

    def test_too_few_keypoints_returns_not_fall(self) -> None:
        pose = empty_pose()
        pose[0] = [100, 50, 0.9]  # only nose visible

        detector = FallDetector(FallRuleConfig(
            min_visible_keypoints=6,
            temporal_window=1,
            temporal_min_fall_votes=1,
        ))
        decision = detector.classify(pose, [50, 50, 200, 300])

        self.assertFalse(decision.is_fall)
        self.assertEqual(decision.reason, "too_few_keypoints")
        self.assertEqual(decision.score, 0.0)

    def test_score_is_zero_when_normal_pose(self) -> None:
        pose = empty_pose()
        pose[5] = [140, 80, 0.9]
        pose[6] = [180, 80, 0.9]
        pose[11] = [145, 220, 0.9]
        pose[12] = [185, 220, 0.9]

        detector = FallDetector(FallRuleConfig(temporal_window=1, temporal_min_fall_votes=1))
        decision = detector.classify(pose, [100, 50, 220, 390])

        self.assertAlmostEqual(decision.score, 0.0, places=2)

    def test_score_threshold_boundary(self) -> None:
        """A pose with score ~0.5 should be fall only if threshold <= score."""
        # Moderate fall pose: horizontal torso but narrow box
        pose = empty_pose()
        pose[5] = [120, 100, 0.9]
        pose[6] = [160, 100, 0.9]
        pose[11] = [200, 102, 0.9]
        pose[12] = [240, 102, 0.9]
        pose[15] = [280, 102, 0.9]
        pose[16] = [320, 102, 0.9]

        detector_very_low = FallDetector(FallRuleConfig(
            fall_score_threshold=0.01,
            temporal_window=1,
            temporal_min_fall_votes=1,
        ))
        detector_very_high = FallDetector(FallRuleConfig(
            fall_score_threshold=0.99,
            temporal_window=1,
            temporal_min_fall_votes=1,
        ))

        decision_low = detector_very_low.classify(pose, [80, 50, 380, 180])
        decision_high = detector_very_high.classify(pose, [80, 50, 380, 180])

        # Very low threshold: always fall (if score > 0)
        self.assertTrue(decision_low.is_fall)
        # Very high threshold: should not be fall unless score >= 0.99
        # (score depends on exact geometry; we just verify the threshold works)
        self.assertEqual(decision_high.is_fall, decision_high.score >= 0.99)

    def test_temporal_voting_requires_min_votes(self) -> None:
        pose = empty_pose()
        pose[5] = [120, 100, 0.9]
        pose[6] = [120, 140, 0.9]
        pose[11] = [260, 105, 0.9]
        pose[12] = [260, 145, 0.9]

        detector = FallDetector(FallRuleConfig(
            temporal_window=5,
            temporal_min_fall_votes=3,
            fall_score_threshold=0.70,
        ))

        d1 = detector.classify(pose, [90, 80, 360, 180])
        self.assertFalse(d1.temporal_is_fall)

    def test_reset_clears_history(self) -> None:
        pose = empty_pose()
        pose[5] = [120, 100, 0.9]
        pose[6] = [120, 140, 0.9]
        pose[11] = [260, 105, 0.9]
        pose[12] = [260, 145, 0.9]

        detector = FallDetector(FallRuleConfig(temporal_window=1, temporal_min_fall_votes=1))
        detector.classify(pose, [90, 80, 360, 180])
        detector.reset()

        normal_pose = empty_pose()
        normal_pose[5] = [140, 80, 0.9]
        normal_pose[6] = [180, 80, 0.9]
        normal_pose[11] = [145, 220, 0.9]
        normal_pose[12] = [185, 220, 0.9]
        decision = detector.classify(normal_pose, [100, 50, 220, 390])
        self.assertFalse(decision.temporal_is_fall)

    def test_gap_score_contributes_to_fall(self) -> None:
        """Small shoulder-hip gap (person lying down) should contribute gap_score."""
        pose = empty_pose()
        pose[5] = [120, 100, 0.9]
        pose[6] = [160, 100, 0.9]
        pose[11] = [200, 102, 0.9]
        pose[12] = [240, 102, 0.9]
        pose[15] = [320, 102, 0.9]
        pose[16] = [360, 102, 0.9]

        detector = FallDetector(FallRuleConfig(
            min_visible_keypoints=6,
            temporal_window=1,
            temporal_min_fall_votes=1,
        ))
        decision = detector.classify(pose, [80, 50, 300, 200])

        self.assertIn("gap=", decision.reason)


if __name__ == "__main__":
    unittest.main()