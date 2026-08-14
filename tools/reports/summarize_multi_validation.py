from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize one or more run_system.py multi-person runs."
    )
    parser.add_argument("--run-dir", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    frames = _read_jsonl(run_dir / "frames.jsonl")
    events = _read_jsonl(run_dir / "events.jsonl")
    detections = [
        detection
        for frame in frames
        for detection in frame.get("detections", [])
    ]
    track_counts = Counter(
        int(detection["track_id"])
        for detection in detections
        if detection.get("track_id") is not None
    )
    frame_count = int(summary.get("frames") or len(frames))
    duration_sec = float(summary.get("duration_sec") or 0.0)

    return {
        "run_name": summary.get("run_name") or run_dir.name,
        "run_dir": str(run_dir),
        "source": summary.get("source"),
        "model": summary.get("model"),
        "image_size": summary.get("args", {}).get("imgsz"),
        "confidence_threshold": summary.get("args", {}).get("conf"),
        "frames": frame_count,
        "duration_sec": duration_sec,
        "observed_end_to_end_fps": (
            round(frame_count / duration_sec, 3) if duration_sec > 0 else None
        ),
        "max_tracked_persons_in_frame": max(
            (int(frame.get("person_count", 0)) for frame in frames),
            default=0,
        ),
        "frames_with_2plus_tracked_persons": sum(
            int(frame.get("person_count", 0)) >= 2 for frame in frames
        ),
        "untracked_detections": sum(
            int(frame.get("untracked_detections", 0)) for frame in frames
        ),
        "unique_track_ids": sorted(track_counts),
        "detections_per_track_id": {
            str(track_id): count for track_id, count in sorted(track_counts.items())
        },
        "event_starts": sum(event.get("type") == "event_start" for event in events),
        "event_clips_saved": sum(
            event.get("type") == "event_clip_saved" for event in events
        ),
        "event_trigger_policy": summary.get("event_trigger_policy"),
        "artifacts": summary.get("artifacts", {}),
    }


def main() -> None:
    args = parse_args()
    payload = {
        "report_type": "multi_person_functional_validation",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "limitations": [
            "Input videos were composited to exercise the multi-person pipeline.",
            "The results are functional evidence, not a real-scene multi-person accuracy benchmark.",
            "Track-ID fragmentation remains visible and requires real-scene tuning.",
            "Observed end-to-end FPS includes decode, tracking, visualization and artifact writing.",
        ],
        "runs": [summarize_run(run_dir) for run_dir in args.run_dir],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
