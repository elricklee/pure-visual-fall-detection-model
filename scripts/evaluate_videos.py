from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fall_detection import FallDetector, FallRuleConfig


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fall alerts on videos.")
    parser.add_argument("--source", default="datasets/fall_pose/videos/raw")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="reports/day2_video_eval.json")
    parser.add_argument("--csv-output", default="reports/day2_video_eval.csv")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--temporal-votes", type=int, default=3)
    parser.add_argument("--fall-score-threshold", type=float, default=0.55)
    parser.add_argument("--bbox-aspect-threshold", type=float, default=1.25)
    parser.add_argument("--torso-angle-threshold", type=float, default=35.0)
    parser.add_argument("--shoulder-hip-gap-threshold", type=float, default=0.22)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def expected_from_name(path: Path) -> bool | None:
    name = path.name.lower()
    if "adl" in name or "normal" in name:
        return False
    if "_fall_" in name or "fall-" in name or name.startswith("fall"):
        return True
    return None


def iter_videos(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def evaluate_video(
    model,
    video_path: Path,
    args: argparse.Namespace,
) -> dict:
    detector = FallDetector(
        FallRuleConfig(
            temporal_window=args.temporal_window,
            temporal_min_fall_votes=args.temporal_votes,
            fall_score_threshold=args.fall_score_threshold,
            bbox_aspect_threshold=args.bbox_aspect_threshold,
            torso_horizontal_angle_threshold=args.torso_angle_threshold,
            shoulder_hip_gap_threshold=args.shoulder_hip_gap_threshold,
        )
    )

    frame_count = 0
    person_frames = 0
    raw_fall_frames = 0
    temporal_fall_frames = 0
    first_alert_frame: int | None = None
    max_score = 0.0

    for result in model.predict(
        source=str(video_path),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    ):
        frame_count += 1
        if args.max_frames > 0 and frame_count > args.max_frames:
            break

        if result.boxes is None or result.keypoints is None:
            detector.reset()
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        if len(boxes) == 0:
            detector.reset()
            continue

        person_frames += 1
        pairs = list(zip(boxes, keypoints))
        pairs.sort(
            key=lambda pair: (pair[0][2] - pair[0][0]) * (pair[0][3] - pair[0][1]),
            reverse=True,
        )
        decision = detector.classify(pairs[0][1], pairs[0][0])
        max_score = max(max_score, decision.score)
        if decision.is_fall:
            raw_fall_frames += 1
        if decision.temporal_is_fall:
            temporal_fall_frames += 1
            if first_alert_frame is None:
                first_alert_frame = frame_count

    expected = expected_from_name(video_path)
    predicted = temporal_fall_frames > 0
    if expected is True and predicted:
        outcome = "TP"
    elif expected is True and not predicted:
        outcome = "FN"
    elif expected is False and predicted:
        outcome = "FP"
    elif expected is False and not predicted:
        outcome = "TN"
    else:
        outcome = "UNKNOWN"

    return {
        "video": str(video_path),
        "expected_fall": expected,
        "predicted_fall": predicted,
        "outcome": outcome,
        "frames": frame_count,
        "person_frames": person_frames,
        "raw_fall_frames": raw_fall_frames,
        "temporal_fall_frames": temporal_fall_frames,
        "first_alert_frame": first_alert_frame,
        "max_score": round(max_score, 3),
    }


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    model = YOLO(args.model)
    videos = iter_videos(Path(args.source))
    if not videos:
        raise SystemExit(f"no videos found in {args.source}")

    rows = [evaluate_video(model, video, args) for video in videos]
    summary = {
        key: sum(row["outcome"] == key for row in rows)
        for key in ("TP", "FP", "TN", "FN", "UNKNOWN")
    }
    total_known = summary["TP"] + summary["FP"] + summary["TN"] + summary["FN"]
    summary["accuracy"] = (
        round((summary["TP"] + summary["TN"]) / total_known, 4)
        if total_known
        else None
    )

    payload = {"summary": summary, "videos": rows}
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    csv_path = Path(args.csv_output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
