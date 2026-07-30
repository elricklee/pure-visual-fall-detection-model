import unittest

from scripts.evaluate_videos import _roc_auc, _temporal_risk_score, _best_threshold


def _row(name: str, expected: bool, max_score: float, frames: int, person_frames: int,
         temporal_fall_frames: int, first_alert_frame: int | None) -> dict:
    row = {
        "video": name,
        "expected_fall": expected,
        "outcome": "TP" if expected else "TN",
        "frames": frames,
        "person_frames": person_frames,
        "temporal_fall_frames": temporal_fall_frames,
        "first_alert_frame": first_alert_frame,
        "max_score": max_score,
    }
    row["temporal_risk_score"] = _temporal_risk_score(row)
    return row


class VideoMetricTest(unittest.TestCase):
    def test_temporal_risk_improves_adl_spike_ranking(self) -> None:
        rows = [
            _row("adl_01.mp4", False, 0.696, 150, 143, 0, None),
            _row("adl_02.mp4", False, 0.955, 180, 169, 13, 9),
            _row("adl_03.mp4", False, 0.354, 180, 179, 0, None),
            _row("adl_04.mp4", False, 0.650, 150, 150, 0, None),
            _row("fall_01.mp4", True, 0.938, 160, 139, 1, 126),
            _row("fall_02.mp4", True, 1.000, 110, 108, 19, 66),
            _row("fall_03.mp4", True, 1.000, 215, 214, 17, 186),
            _row("fall_04.mp4", True, 1.000, 96, 93, 40, 42),
        ]

        self.assertEqual(_roc_auc(rows, "max_score"), 0.9375)
        self.assertEqual(_roc_auc(rows, "temporal_risk_score"), 1.0)

    def test_auc_is_none_without_both_classes(self) -> None:
        rows = [
            _row("fall_01.mp4", True, 0.9, 100, 100, 10, 50),
            _row("fall_02.mp4", True, 0.8, 100, 100, 8, 40),
        ]

        self.assertIsNone(_roc_auc(rows, "temporal_risk_score"))

    def test_best_threshold_returns_operating_point(self) -> None:
        rows = [
            _row("adl.mp4", False, 0.9, 100, 100, 0, None),
            _row("fall.mp4", True, 0.9, 100, 100, 10, 50),
        ]

        summary = _best_threshold(rows, "temporal_risk_score")

        self.assertIsNotNone(summary)
        self.assertEqual(summary["TP"], 1)
        self.assertEqual(summary["FP"], 0)
        self.assertEqual(summary["balanced_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
