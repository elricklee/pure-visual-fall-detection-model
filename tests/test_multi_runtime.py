from types import SimpleNamespace
import unittest

from scripts.evaluation.evaluate_videos import _summarize_multi_frame_metrics
from scripts.runtime.run_system import _should_emit_fall_event


def _person(score: float, raw_fall: bool, confirmed_fall: bool):
    decision = SimpleNamespace(
        score=score,
        is_fall=raw_fall,
        temporal_is_fall=confirmed_fall,
    )
    return SimpleNamespace(decision=decision)


class MultiFrameMetricTest(unittest.TestCase):
    def test_two_falling_people_count_as_one_video_frame(self) -> None:
        results = [
            _person(0.81, True, True),
            _person(0.93, True, True),
        ]

        metrics = _summarize_multi_frame_metrics(
            results,
            raw_fall_frames=4,
            temporal_fall_frames=2,
            first_alert_frame=None,
            max_score=0.5,
            frame_count=10,
        )

        self.assertEqual(metrics, (1, 5, 3, 10, 0.93))

    def test_empty_frame_does_not_change_counts(self) -> None:
        metrics = _summarize_multi_frame_metrics(
            [],
            raw_fall_frames=4,
            temporal_fall_frames=2,
            first_alert_frame=7,
            max_score=0.8,
            frame_count=10,
        )
        self.assertEqual(metrics, (0, 4, 2, 7, 0.8))


class FallEventTriggerTest(unittest.TestCase):
    def test_continuous_confirmed_state_emits_once(self) -> None:
        last_event_t: dict[int, float] = {}
        active: set[int] = set()

        first = _should_emit_fall_event(1, True, 1.0, last_event_t, active, 10.0)
        self.assertTrue(first)
        last_event_t[1] = 1.0

        self.assertFalse(
            _should_emit_fall_event(1, True, 20.0, last_event_t, active, 10.0)
        )

    def test_recovered_person_can_emit_new_incident_after_cooldown(self) -> None:
        last_event_t = {1: 1.0}
        active = {1}

        self.assertFalse(
            _should_emit_fall_event(1, False, 5.0, last_event_t, active, 10.0)
        )
        self.assertTrue(
            _should_emit_fall_event(1, True, 12.0, last_event_t, active, 10.0)
        )

    def test_recurrence_inside_cooldown_is_suppressed(self) -> None:
        last_event_t = {1: 5.0}
        active: set[int] = set()
        self.assertFalse(
            _should_emit_fall_event(1, True, 8.0, last_event_t, active, 10.0)
        )
        self.assertIn(1, active)


if __name__ == "__main__":
    unittest.main()
