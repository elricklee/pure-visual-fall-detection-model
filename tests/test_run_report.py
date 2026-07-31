import json
import tempfile
import unittest
from pathlib import Path

from scripts.runtime.generate_run_report import generate_report


class GenerateRunReportTest(unittest.TestCase):
    def test_generates_html_with_event_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            event_dir = run_dir / "events" / "demo_p1_000010"
            event_dir.mkdir(parents=True)
            (event_dir / "snapshot.jpg").write_bytes(b"fake")
            (event_dir / "replay.mp4").write_bytes(b"fake")
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "run_name": "demo",
                        "fps": 25.0,
                        "frames": 100,
                        "events": 1,
                        "duration_sec": 4.0,
                        "width": 640,
                        "height": 480,
                        "started_at": "2026-07-28T10:00:00",
                        "artifacts": {"live_video": str(run_dir / "live.mp4")},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "events.jsonl").write_text(
                json.dumps(
                    {
                        "type": "event_start",
                        "alert": {
                            "event_id": "demo_p1_000010",
                            "track_id": 1,
                            "state": "FALL_CONFIRMED",
                            "score": 0.91,
                            "reason": "bbox=1.00+torso=0.80",
                            "frame_index": 10,
                            "t_sec": 0.4,
                            "snapshot_path": str(event_dir / "snapshot.jpg"),
                            "clip_path": str(event_dir / "replay.mp4"),
                        },
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "type": "event_clip_saved",
                        "event_id": "demo_p1_000010",
                        "clip_path": str(event_dir / "replay.mp4"),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "frames.jsonl").write_text(
                json.dumps(
                    {
                        "type": "frame_trace",
                        "frame_index": 10,
                        "t_sec": 0.4,
                        "active_ids": [1],
                        "person_count": 1,
                        "detections": [
                            {
                                "track_id": 1,
                                "det_conf": 0.85,
                                "score": 0.91,
                                "is_fall": True,
                                "temporal_is_fall": True,
                                "state": "FALL_CONFIRMED",
                                "reason": "bbox=1.00+torso=0.80",
                                "box_xyxy": [100, 100, 300, 300],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report_path = generate_report(run_dir)
            html = report_path.read_text(encoding="utf-8")

            self.assertIn("Fall Alert Run Report", html)
            self.assertIn("demo_p1_000010", html)
            self.assertIn("events/demo_p1_000010/snapshot.jpg", html)
            self.assertIn("events/demo_p1_000010/replay.mp4", html)
            self.assertIn("Frame Trace", html)
            self.assertIn("frames.jsonl", html)
            self.assertIn("#1 FALL_CONFIRMED det=0.85 score=0.91", html)


if __name__ == "__main__":
    unittest.main()
