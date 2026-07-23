import unittest

import numpy as np

from fall_detection import FallDetector, FallRuleConfig


def empty_pose() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


class FallLogicTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
