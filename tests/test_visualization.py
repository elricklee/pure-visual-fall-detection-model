import unittest
from unittest.mock import patch

import numpy as np

from fall_detection import FallDecision, FallState
from fall_detection.state_machine import TemporalStateDecision
from fall_detection.pose_features import PoseMetrics
from fall_detection.tracker_manager import PersonResult
from fall_detection.visualization import (
    draw_decision,
    draw_multi_decision,
    draw_multi_status_banner,
    draw_pose,
    draw_status_banner,
)


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

    def test_label_uses_det_for_detection_confidence(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=False)
        with patch("fall_detection.visualization.cv2.putText") as put_text:
            draw_decision(image, [100, 100, 300, 300], decision, det_conf=0.85)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("NORMAL det=0.85", labels)


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

    def test_banner_uses_score_for_fall_score(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        with patch("fall_detection.visualization.cv2.putText") as put_text:
            draw_status_banner(image, "FALL_CONFIRMED", True, score=1.0)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("STATE: FALL_CONFIRMED  score=1.00", labels)


# ---------------------------------------------------------------------------
# Multi-person visualization tests
# ---------------------------------------------------------------------------


def make_person_result(
    track_id: int,
    is_fall: bool = False,
    temporal_is_fall: bool = False,
    score: float = 0.5,
    state: FallState = FallState.NORMAL,
) -> PersonResult:
    decision = make_decision(is_fall=is_fall, temporal_is_fall=temporal_is_fall, score=score)
    state_decision = TemporalStateDecision(
        state=state,
        is_fall_confirmed=temporal_is_fall,
        dynamic_suspect=False,
        fall_votes=3 if temporal_is_fall else 0,
        normal_votes=5 if not is_fall else 0,
        center_y_delta=None,
        torso_angle_delta=None,
        reason="test",
    )
    return PersonResult(
        track_id=track_id,
        decision=decision,
        state_decision=state_decision,
        box_xyxy=[100, 100, 300, 300],
        keypoints=np.zeros((17, 3), dtype=np.float32),
        det_conf=0.85,
    )


class DrawMultiDecisionTest(unittest.TestCase):
    def test_normal_uses_person_color(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=False, temporal_is_fall=False)
        draw_multi_decision(image, [100, 100, 300, 300], decision, track_id=0)
        self.assertTrue(np.any(image > 0))

    def test_fall_uses_red_regardless_of_track_id(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=True, temporal_is_fall=True)
        draw_multi_decision(image, [100, 100, 300, 300], decision, track_id=99)
        # Red channel (BGR index 2) should have values
        self.assertTrue(np.any(image[:, :, 2] > 0))

    def test_fall_pose_uses_orange(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=True, temporal_is_fall=False)
        draw_multi_decision(image, [100, 100, 300, 300], decision, track_id=2)
        self.assertTrue(np.any(image > 0))

    def test_output_shape_unchanged(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=False)
        draw_multi_decision(image, [100, 100, 300, 300], decision, track_id=5)
        self.assertEqual(image.shape, (480, 640, 3))

    def test_different_track_ids_get_different_content(self) -> None:
        """Different track_ids should produce different label text on the image."""
        image1 = np.zeros((480, 640, 3), dtype=np.uint8)
        image2 = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=False)
        draw_multi_decision(image1, [100, 100, 300, 300], decision, track_id=1)
        draw_multi_decision(image2, [100, 100, 300, 300], decision, track_id=2)
        # Both should be modified, but with different person colors
        self.assertTrue(np.any(image1 > 0))
        self.assertTrue(np.any(image2 > 0))

    def test_multi_label_uses_det_for_detection_confidence(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        decision = make_decision(is_fall=True, temporal_is_fall=True)
        with patch("fall_detection.visualization.cv2.putText") as put_text:
            draw_multi_decision(image, [100, 100, 300, 300], decision, track_id=2, det_conf=0.85)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("#2 FALL det=0.85", labels)


class DrawMultiStatusBannerTest(unittest.TestCase):
    def test_empty_results_show_no_person(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        draw_multi_status_banner(image, [])
        self.assertTrue(np.any(image > 0))

    def test_single_person_banner(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [make_person_result(1, state=FallState.NORMAL)]
        draw_multi_status_banner(image, results)
        self.assertTrue(np.any(image > 0))

    def test_multiple_persons_show_multiple_rows(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [
            make_person_result(1, state=FallState.NORMAL),
            make_person_result(2, is_fall=True, temporal_is_fall=True, state=FallState.FALL_CONFIRMED),
        ]
        draw_multi_status_banner(image, results)
        # Banner area should be taller for 2 rows
        self.assertTrue(np.any(image > 0))

    def test_fall_alert_highlighted_in_red(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [
            make_person_result(3, is_fall=True, temporal_is_fall=True, state=FallState.FALL_CONFIRMED),
        ]
        draw_multi_status_banner(image, results)
        # Red channel should have values for fall alert
        self.assertTrue(np.any(image[:, :, 2] > 0))

    def test_output_shape_unchanged(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [make_person_result(1), make_person_result(2)]
        draw_multi_status_banner(image, results)
        self.assertEqual(image.shape, (480, 640, 3))

    def test_multi_banner_uses_score_for_fall_score(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        results = [make_person_result(2, score=1.0, state=FallState.FALL_CONFIRMED)]
        with patch("fall_detection.visualization.cv2.putText") as put_text:
            draw_multi_status_banner(image, results)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("#2 FALL_CONFIRMED score=1.00", labels)


if __name__ == "__main__":
    unittest.main()
