import unittest

import numpy as np

from fall_detection import FallDecision
from fall_detection.pose_features import PoseMetrics
from fall_detection.visualization import draw_decision, draw_pose, draw_status_banner


def make_decision(is_fall: bool, temporal_is_fall: bool = False, score: float = 0.5) -> FallDecision:
    return FallDecision(
        is_fall=is_fall,
        score=score,
        reason="test",
        metrics=PoseMetrics(
            bbox_aspect=1.0,
            torso_angle_from_horizontal=45.0,
            shoulder_hip_vertical_gap_ratio=0.3,
            visible_keypoints=17,
        ),
        temporal_is_fall=temporal_is_fall,
    )


class DrawPoseTest(unittest.TestCase):
    def test_draws_without_error(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[5] = [140, 80, 0.9]
        kpts[6] = [180, 80, 0.9]
        kpts[11] = [145, 220, 0.9]
        kpts[12] = [185, 220, 0.9]

        draw_pose(image, kpts)
        # Should not raise, and image should be modified
        self.assertTrue(np.any(image > 0))

    def test_low_confidence_keypoints_skipped(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = np.zeros((17, 3), dtype=np.float32)
        # All keypoints have zero confidence
        draw_pose(image, kpts)
        self.assertFalse(np.any(image > 0))

    def test_output_shape_unchanged(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        kpts = np.zeros((17, 3), dtype=np.float32)
        kpts[5] = [140, 80, 0.9]
        draw_pose(image, kpts)
        self.assertEqual(image.shape, (480, 640, 3))


class DrawDecisionTest(unittest.TestCase):
    def test_fall_confirmed_draws_red(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=True, temporal_is_fall=True)
        draw_decision(image, [100, 100, 300, 300], decision)
        # Red channel should have values (BGR format: [0,0,255] = red)
        self.assertTrue(np.any(image[:, :, 2] > 0))

    def test_fall_pose_draws_orange(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=True, temporal_is_fall=False)
        draw_decision(image, [100, 100, 300, 300], decision)
        # Orange = (0, 165, 255) in BGR
        self.assertTrue(np.any(image > 0))

    def test_normal_draws_green(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=False, temporal_is_fall=False)
        draw_decision(image, [100, 100, 300, 300], decision)
        # Green = (0, 180, 0) in BGR
        self.assertTrue(np.any(image[:, :, 1] > 0))

    def test_output_shape_unchanged(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=False)
        draw_decision(image, [100, 100, 300, 300], decision)
        self.assertEqual(image.shape, (480, 640, 3))


class DrawStatusBannerTest(unittest.TestCase):
    def test_alert_banner(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_status_banner(image, "FALL_CONFIRMED", True)
        # Banner area should be modified
        self.assertTrue(np.any(image[:50, :320] > 0))

    def test_normal_banner(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_status_banner(image, "NORMAL", False)
        self.assertTrue(np.any(image[:50, :320] > 0))

    def test_output_shape_unchanged(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_status_banner(image, "NORMAL", False)
        self.assertEqual(image.shape, (480, 640, 3))


if __name__ == "__main__":
    unittest.main()