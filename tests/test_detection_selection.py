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


if __name__ == "__main__":
    unittest.main()
