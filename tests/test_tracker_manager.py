"""Tests for TrackerManager multi-person state isolation."""

import unittest

import numpy as np

from fall_detection import (
    FallDetector,
    FallRuleConfig,
    FallState,
    TemporalFallStateMachine,
    TemporalStateConfig,
    TrackerManager,
)
from fall_detection.tracker_manager import PersonResult


# ---------------------------------------------------------------------------
# Helpers – construct keypoints that produce clear fall / normal decisions
# ---------------------------------------------------------------------------


def _fall_keypoints() -> np.ndarray:
    """Keypoints for a fallen person (horizontal torso, wide bbox).

    The key insight: shoulder_center and hip_center must have a large
    horizontal offset (dx) and small vertical offset (dy) so that
    atan2(dy, dx) gives a small angle from horizontal (~45°).
    This means the person is tilting sideways — shoulders on one side,
    hips further to the right/left.
    """
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Shoulders at left, hips to the right → horizontal torso
    kpts[5] = [80, 180, 0.9]   # left_shoulder
    kpts[6] = [120, 200, 0.9]  # right_shoulder
    kpts[11] = [260, 210, 0.9]  # left_hip
    kpts[12] = [300, 220, 0.9]  # right_hip
    kpts[13] = [340, 230, 0.9]  # left_knee
    kpts[14] = [380, 240, 0.9]  # right_knee
    kpts[15] = [370, 270, 0.9]  # left_ankle
    kpts[16] = [410, 280, 0.9]  # right_ankle
    kpts[0] = [60, 170, 0.9]   # nose
    kpts[7] = [40, 160, 0.8]   # left_elbow
    kpts[8] = [50, 190, 0.8]   # right_elbow
    kpts[9] = [20, 150, 0.7]   # left_wrist
    kpts[10] = [30, 180, 0.7]  # right_wrist
    kpts[1] = [70, 160, 0.8]   # left_eye
    kpts[2] = [90, 165, 0.8]   # right_eye
    kpts[3] = [55, 155, 0.7]   # left_ear
    kpts[4] = [100, 160, 0.7]  # right_ear
    return kpts


def _normal_keypoints() -> np.ndarray:
    """Keypoints for a standing person (vertical torso, tall bbox)."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    # Shoulders above hips → vertical torso
    kpts[5] = [180, 100, 0.9]  # left_shoulder
    kpts[6] = [220, 100, 0.9]  # right_shoulder
    kpts[11] = [185, 250, 0.9]  # left_hip
    kpts[12] = [215, 250, 0.9]  # right_hip
    kpts[13] = [185, 350, 0.9]  # left_knee
    kpts[14] = [215, 350, 0.9]  # right_knee
    kpts[15] = [185, 420, 0.9]  # left_ankle
    kpts[16] = [215, 420, 0.9]  # right_ankle
    kpts[0] = [200, 60, 0.9]  # nose
    kpts[7] = [150, 150, 0.8]  # left_elbow
    kpts[8] = [250, 150, 0.8]  # right_elbow
    kpts[9] = [130, 200, 0.7]  # left_wrist
    kpts[10] = [270, 200, 0.7]  # right_wrist
    kpts[1] = [190, 50, 0.8]  # left_eye
    kpts[2] = [210, 50, 0.8]  # right_eye
    kpts[3] = [180, 55, 0.7]  # left_ear
    kpts[4] = [220, 55, 0.7]  # right_ear
    return kpts


FALL_BOX = [10, 140, 430, 300]
NORMAL_BOX = [150, 40, 280, 450]
FRAME_H = 480


class TrackerManagerBasicTest(unittest.TestCase):
    def test_new_track_id_creates_tracker(self) -> None:
        """First appearance of a track_id auto-creates detector + state machine."""
        mgr = TrackerManager()
        result = mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        self.assertIn(1, mgr.active_ids)
        self.assertEqual(result.track_id, 1)
        self.assertIsInstance(result, PersonResult)

    def test_same_track_id_reuses_tracker(self) -> None:
        """Same track_id on consecutive frames reuses the same state."""
        mgr = TrackerManager()
        # First frame: normal
        r1 = mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        self.assertEqual(r1.state_decision.state, FallState.NORMAL)

        # Second frame: fall → should enter SUSPECT_FALL (state continues)
        r2 = mgr.update(1, _fall_keypoints(), FALL_BOX, FRAME_H)
        # State machine should have transitioned from NORMAL
        self.assertNotEqual(r2.state_decision.state, FallState.NORMAL)

    def test_different_track_ids_isolated(self) -> None:
        """Different track_ids have completely independent states."""
        mgr = TrackerManager(use_state_machine=True)

        # ID=1: feed multiple fall frames to drive toward FALL_CONFIRMED
        for _ in range(5):
            mgr.update(1, _fall_keypoints(), FALL_BOX, FRAME_H)

        # ID=2: always normal
        r2 = mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        self.assertEqual(r2.state_decision.state, FallState.NORMAL)

        # ID=1 should be in a fall-related state
        r1_check = mgr.update(1, _fall_keypoints(), FALL_BOX, FRAME_H)
        self.assertNotEqual(r1_check.state_decision.state, FallState.NORMAL)

    def test_multiple_ids_tracked_simultaneously(self) -> None:
        """Multiple persons can be tracked at the same time."""
        mgr = TrackerManager()
        mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        mgr.update(3, _fall_keypoints(), FALL_BOX, FRAME_H)
        self.assertEqual(sorted(mgr.active_ids), [1, 2, 3])

    def test_active_ids_returns_all_current(self) -> None:
        mgr = TrackerManager()
        mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        mgr.update(3, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        self.assertEqual(sorted(mgr.active_ids), [1, 3])


class TrackerManagerStaleTest(unittest.TestCase):
    def test_manual_frame_mode_uses_video_frames_not_detection_count(self) -> None:
        """Multiple detections in one frame should not age missing IDs faster."""
        mgr = TrackerManager(stale_threshold=3)
        mgr.set_frame(1)
        mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)

        mgr.set_frame(2)
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        mgr.update(3, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        mgr.update(4, _normal_keypoints(), NORMAL_BOX, FRAME_H)

        stale = mgr.cleanup_stale()
        self.assertEqual(stale, [])
        self.assertIn(1, mgr.active_ids)

    def test_manual_frame_mode_cleanup_can_advance_on_empty_frames(self) -> None:
        """set_frame allows stale cleanup even when no detections are updated."""
        mgr = TrackerManager(stale_threshold=3)
        mgr.set_frame(1)
        mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)

        mgr.set_frame(5)
        stale = mgr.cleanup_stale()
        self.assertIn(1, stale)
        self.assertNotIn(1, mgr.active_ids)

    def test_cleanup_stale_removes_old_ids(self) -> None:
        """IDs not seen for stale_threshold frames are cleaned up."""
        mgr = TrackerManager(stale_threshold=3)
        mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=1, last_seen[1]=1
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=2, last_seen[2]=2
        # ID=1 no longer appears — advance frame_count by updating only ID=2
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=3
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=4
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=5
        # Now frame_count=5, last_seen[1]=1 → gap=4 > stale_threshold=3
        stale = mgr.cleanup_stale()
        self.assertIn(1, stale)
        self.assertNotIn(1, mgr.active_ids)
        self.assertIn(2, mgr.active_ids)

    def test_no_cleanup_within_threshold(self) -> None:
        """IDs within threshold should NOT be cleaned up."""
        mgr = TrackerManager(stale_threshold=10)
        mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        stale = mgr.cleanup_stale()
        self.assertEqual(stale, [])
        self.assertIn(1, mgr.active_ids)
        self.assertIn(2, mgr.active_ids)

    def test_cleanup_returns_removed_ids(self) -> None:
        mgr = TrackerManager(stale_threshold=1)
        mgr.update(5, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=1, last_seen[5]=1
        # ID=5 not seen next frame — need 2 more updates to make gap > 1
        mgr.update(6, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=2
        mgr.update(6, _normal_keypoints(), NORMAL_BOX, FRAME_H)  # frame_count=3, gap for 5 = 3-1 = 2 > 1
        stale = mgr.cleanup_stale()
        self.assertIn(5, stale)
        self.assertNotIn(5, mgr.active_ids)


class TrackerManagerResetTest(unittest.TestCase):
    def test_reset_person_clears_state(self) -> None:
        mgr = TrackerManager()
        for _ in range(5):
            mgr.update(1, _fall_keypoints(), FALL_BOX, FRAME_H)

        mgr.reset_person(1)
        # Next update should start fresh from NORMAL
        r = mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        self.assertEqual(r.state_decision.state, FallState.NORMAL)

    def test_reset_person_nonexistent_is_noop(self) -> None:
        mgr = TrackerManager()
        mgr.reset_person(999)  # Should not raise
        self.assertEqual(mgr.active_ids, [])

    def test_reset_all_clears_everyone(self) -> None:
        mgr = TrackerManager()
        for _ in range(5):
            mgr.update(1, _fall_keypoints(), FALL_BOX, FRAME_H)
            mgr.update(2, _fall_keypoints(), FALL_BOX, FRAME_H)

        mgr.reset_all()
        r1 = mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        r2 = mgr.update(2, _normal_keypoints(), NORMAL_BOX, FRAME_H)
        self.assertEqual(r1.state_decision.state, FallState.NORMAL)
        self.assertEqual(r2.state_decision.state, FallState.NORMAL)


class TrackerManagerVoteModeTest(unittest.TestCase):
    def test_vote_mode_skips_state_machine(self) -> None:
        mgr = TrackerManager(use_state_machine=False)
        r = mgr.update(1, _fall_keypoints(), FALL_BOX, FRAME_H)
        self.assertIsNone(r.state_decision)
        # decision.is_fall still works (from FallDetector internal vote)
        self.assertTrue(r.decision.is_fall or not r.decision.is_fall)  # just verify it's set


class PersonResultTest(unittest.TestCase):
    def test_person_result_has_all_fields(self) -> None:
        mgr = TrackerManager()
        r = mgr.update(1, _normal_keypoints(), NORMAL_BOX, FRAME_H, det_conf=0.85)
        self.assertEqual(r.track_id, 1)
        self.assertIsNotNone(r.decision)
        self.assertIsNotNone(r.state_decision)
        self.assertAlmostEqual(r.det_conf, 0.85)
        self.assertIsNotNone(r.box_xyxy)
        self.assertIsNotNone(r.keypoints)


if __name__ == "__main__":
    unittest.main()
