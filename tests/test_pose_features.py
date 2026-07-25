import unittest

import numpy as np

from fall_detection.pose_features import (
    PoseMetrics,
    as_keypoint_array,
    bbox_aspect_ratio,
    estimate_pose_metrics,
    midpoint,
    visible_point,
)


def empty_pose() -> np.ndarray:
    return np.zeros((17, 3), dtype=np.float32)


class AsKeypointArrayTest(unittest.TestCase):
    def test_shape_17x3(self) -> None:
        kpts = np.random.rand(17, 3).astype(np.float32)
        result = as_keypoint_array(kpts)
        self.assertEqual(result.shape, (17, 3))

    def test_shape_17x2_adds_confidence(self) -> None:
        kpts = np.random.rand(17, 2).astype(np.float32)
        result = as_keypoint_array(kpts)
        self.assertEqual(result.shape, (17, 3))
        np.testing.assert_array_equal(result[:, :2], kpts)
        np.testing.assert_array_equal(result[:, 2], 1.0)

    def test_wrong_shape_raises(self) -> None:
        with self.assertRaises(ValueError):
            as_keypoint_array(np.zeros((5, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            as_keypoint_array(np.zeros((17, 5), dtype=np.float32))


class VisiblePointTest(unittest.TestCase):
    def test_returns_point_when_confident(self) -> None:
        kpts = empty_pose()
        kpts[5] = [100, 200, 0.9]
        result = visible_point(kpts, "left_shoulder", 0.25)
        self.assertIsNotNone(result)
        np.testing.assert_array_almost_equal(result, [100, 200])

    def test_returns_none_when_low_confidence(self) -> None:
        kpts = empty_pose()
        kpts[5] = [100, 200, 0.1]
        result = visible_point(kpts, "left_shoulder", 0.25)
        self.assertIsNone(result)

    def test_returns_none_when_zero_confidence(self) -> None:
        kpts = empty_pose()
        result = visible_point(kpts, "left_shoulder", 0.25)
        self.assertIsNone(result)


class MidpointTest(unittest.TestCase):
    def test_two_points(self) -> None:
        p1 = np.array([0.0, 0.0])
        p2 = np.array([100.0, 200.0])
        result = midpoint([p1, p2])
        np.testing.assert_array_almost_equal(result, [50.0, 100.0])

    def test_none_points_ignored(self) -> None:
        p1 = np.array([100.0, 200.0])
        result = midpoint([None, p1, None])
        np.testing.assert_array_almost_equal(result, [100.0, 200.0])

    def test_all_none_returns_none(self) -> None:
        self.assertIsNone(midpoint([None, None]))


class BboxAspectRatioTest(unittest.TestCase):
    def test_tall_box(self) -> None:
        result = bbox_aspect_ratio([10, 10, 50, 210])
        self.assertAlmostEqual(result, 40 / 200)

    def test_wide_box(self) -> None:
        result = bbox_aspect_ratio([10, 10, 210, 50])
        self.assertAlmostEqual(result, 200 / 40)

    def test_square_box(self) -> None:
        result = bbox_aspect_ratio([10, 10, 110, 110])
        self.assertAlmostEqual(result, 1.0)

    def test_none_returns_none(self) -> None:
        self.assertIsNone(bbox_aspect_ratio(None))

    def test_zero_height_returns_none(self) -> None:
        self.assertIsNone(bbox_aspect_ratio([10, 100, 200, 100]))


class EstimatePoseMetricsTest(unittest.TestCase):
    def test_standing_person_vertical_torso(self) -> None:
        pose = empty_pose()
        pose[5] = [140, 80, 0.9]   # left shoulder
        pose[6] = [180, 80, 0.9]   # right shoulder
        pose[11] = [145, 220, 0.9] # left hip
        pose[12] = [185, 220, 0.9] # right hip

        metrics = estimate_pose_metrics(pose, [100, 50, 220, 390])
        self.assertIsNotNone(metrics.torso_angle_from_horizontal)
        self.assertGreater(metrics.torso_angle_from_horizontal, 70.0)  # standing ≈ 90°
        self.assertGreaterEqual(metrics.visible_keypoints, 4)

    def test_fallen_person_horizontal_torso(self) -> None:
        pose = empty_pose()
        pose[5] = [120, 100, 0.9]  # left shoulder
        pose[6] = [120, 140, 0.9]  # right shoulder
        pose[11] = [260, 105, 0.9] # left hip
        pose[12] = [260, 145, 0.9] # right hip

        metrics = estimate_pose_metrics(pose, [90, 80, 360, 180])
        self.assertIsNotNone(metrics.torso_angle_from_horizontal)
        self.assertLess(metrics.torso_angle_from_horizontal, 30.0)  # lying flat ≈ 0°

    def test_wide_box_high_aspect(self) -> None:
        metrics = estimate_pose_metrics(empty_pose(), [10, 100, 410, 200])
        self.assertIsNotNone(metrics.bbox_aspect)
        self.assertGreater(metrics.bbox_aspect, 1.0)

    def test_tall_box_low_aspect(self) -> None:
        metrics = estimate_pose_metrics(empty_pose(), [100, 10, 200, 410])
        self.assertIsNotNone(metrics.bbox_aspect)
        self.assertLess(metrics.bbox_aspect, 1.0)

    def test_visible_keypoints_count(self) -> None:
        pose = empty_pose()
        for i in range(8):
            pose[i] = [i * 10, i * 10, 0.9]
        metrics = estimate_pose_metrics(pose)
        self.assertEqual(metrics.visible_keypoints, 8)

    def test_shoulder_hip_gap_ratio(self) -> None:
        pose = empty_pose()
        pose[5] = [140, 80, 0.9]
        pose[6] = [180, 80, 0.9]
        pose[11] = [145, 220, 0.9]
        pose[12] = [185, 220, 0.9]

        metrics = estimate_pose_metrics(pose, [100, 50, 220, 390])
        self.assertIsNotNone(metrics.shoulder_hip_vertical_gap_ratio)
        self.assertGreater(metrics.shoulder_hip_vertical_gap_ratio, 0.0)

    def test_no_box_no_aspect_no_gap(self) -> None:
        pose = empty_pose()
        pose[5] = [140, 80, 0.9]
        pose[6] = [180, 80, 0.9]
        pose[11] = [145, 220, 0.9]
        pose[12] = [185, 220, 0.9]

        metrics = estimate_pose_metrics(pose)
        self.assertIsNone(metrics.bbox_aspect)
        self.assertIsNone(metrics.shoulder_hip_vertical_gap_ratio)
        self.assertIsNotNone(metrics.torso_angle_from_horizontal)


if __name__ == "__main__":
    unittest.main()