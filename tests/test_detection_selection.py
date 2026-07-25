import unittest

import numpy as np

from fall_detection.detection_selection import select_primary_pose


def pose(visible_points: int) -> np.ndarray:
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:visible_points, 2] = 0.9
    return keypoints


class DetectionSelectionTest(unittest.TestCase):
    def test_auto_roi_selects_right_side_in_wide_video(self) -> None:
        boxes = [
            [20, 20, 380, 360],
            [500, 40, 750, 350],
        ]
        keypoints = [pose(17), pose(12)]

        selected = select_primary_pose(boxes, keypoints, 800, 400, "auto")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[1], dtype=np.float32))

    def test_full_roi_selects_largest_person(self) -> None:
        boxes = [
            [20, 20, 180, 300],
            [220, 40, 500, 360],
        ]
        keypoints = [pose(17), pose(10)]

        selected = select_primary_pose(boxes, keypoints, 640, 480, "full")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[1], dtype=np.float32))

    def test_left_roi_selects_left_person(self) -> None:
        boxes = [
            [20, 20, 180, 300],
            [500, 40, 750, 350],
        ]
        keypoints = [pose(17), pose(17)]

        selected = select_primary_pose(boxes, keypoints, 800, 400, "left")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[0], dtype=np.float32))

    def test_right_roi_selects_right_person(self) -> None:
        boxes = [
            [20, 20, 180, 300],
            [500, 40, 750, 350],
        ]
        keypoints = [pose(17), pose(17)]

        selected = select_primary_pose(boxes, keypoints, 800, 400, "right")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[1], dtype=np.float32))

    def test_no_person_returns_none(self) -> None:
        selected = select_primary_pose([], [], 640, 480, "full")
        self.assertIsNone(selected)

    def test_auto_narrow_video_uses_full(self) -> None:
        """Narrow video (aspect < 1.8) should default to full ROI."""
        boxes = [[100, 50, 300, 400]]
        keypoints = [pose(17)]

        selected = select_primary_pose(boxes, keypoints, 640, 480, "auto")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[0], dtype=np.float32))

    def test_fallback_to_full_when_no_one_in_roi(self) -> None:
        """If no one is in the right half of a wide video, fall back to full."""
        boxes = [[20, 20, 200, 300]]  # person in left half only
        keypoints = [pose(17)]

        selected = select_primary_pose(boxes, keypoints, 800, 400, "auto")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[0], dtype=np.float32))

    def test_single_person_in_right_half(self) -> None:
        boxes = [[500, 40, 750, 350]]
        keypoints = [pose(17)]

        selected = select_primary_pose(boxes, keypoints, 800, 400, "auto")

        self.assertIsNotNone(selected)
        box, _ = selected
        np.testing.assert_array_equal(box, np.asarray(boxes[0], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()