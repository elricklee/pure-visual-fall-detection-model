from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two evaluate_videos.py JSON reports."
    )
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))

    reference_rows = {
        Path(row["video"]).name: row for row in reference["videos"]
    }
    candidate_rows = {
        Path(row["video"]).name: row for row in candidate["videos"]
    }
    common = sorted(reference_rows.keys() & candidate_rows.keys())
    if not common:
        raise SystemExit("reports have no common videos")

    rows = []
    for name in common:
        left = reference_rows[name]
        right = candidate_rows[name]
        left_alert = left["first_alert_frame"]
        right_alert = right["first_alert_frame"]
        first_alert_delta = (
            abs(left_alert - right_alert)
            if left_alert is not None and right_alert is not None
            else None
        )
        rows.append(
            {
                "video": name,
                "predicted_fall_match": (
                    left["predicted_fall"] == right["predicted_fall"]
                ),
                "frame_count_match": left["frames"] == right["frames"],
                "person_frame_delta": right["person_frames"] - left["person_frames"],
                "temporal_fall_frame_delta": (
                    right["temporal_fall_frames"] - left["temporal_fall_frames"]
                ),
                "first_alert_frame_delta": first_alert_delta,
            }
        )

    alert_deltas = [
        row["first_alert_frame_delta"]
        for row in rows
        if row["first_alert_frame_delta"] is not None
    ]
    payload = {
        "mode": args.mode,
        "reference_report": str(args.reference.resolve()),
        "candidate_report": str(args.candidate.resolve()),
        "videos": len(rows),
        "predicted_fall_agreement": round(
            sum(row["predicted_fall_match"] for row in rows) / len(rows),
            4,
        ),
        "frame_count_agreement": round(
            sum(row["frame_count_match"] for row in rows) / len(rows),
            4,
        ),
        "mean_absolute_person_frame_delta": round(
            sum(abs(row["person_frame_delta"]) for row in rows) / len(rows),
            4,
        ),
        "mean_absolute_temporal_fall_frame_delta": round(
            sum(abs(row["temporal_fall_frame_delta"]) for row in rows) / len(rows),
            4,
        ),
        "mean_absolute_first_alert_frame_delta": (
            round(sum(alert_deltas) / len(alert_deltas), 4)
            if alert_deltas
            else None
        ),
        "ground_truth_status": (
            "unavailable: all current video filenames map to UNKNOWN; "
            "agreement is not accuracy"
        ),
        "details": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
