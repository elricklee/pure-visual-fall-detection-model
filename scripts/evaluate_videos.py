from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fall_detection import (
    FallDetector,
    FallRuleConfig,
    TemporalFallStateMachine,
    TemporalStateConfig,
    TrackerManager,
)
from fall_detection.detection_selection import select_primary_pose
from fall_detection.image_utils import to_infrared


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fall alerts on videos.")
    parser.add_argument("--source", default="datasets/fall_pose/videos/raw")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="reports/evaluation/video_evaluation.json")
    parser.add_argument("--csv-output", default="reports/evaluation/video_evaluation.csv")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--temporal-votes", type=int, default=3)
    parser.add_argument("--fall-score-threshold", type=float, default=0.70)
    parser.add_argument("--bbox-aspect-threshold", type=float, default=1.25)
    parser.add_argument("--torso-angle-threshold", type=float, default=35.0)
    parser.add_argument("--shoulder-hip-gap-threshold", type=float, default=0.22)
    parser.add_argument(
        "--decision-mode",
        choices=["state_machine", "vote"],
        default="state_machine",
    )
    parser.add_argument(
        "--roi",
        choices=["auto", "full", "left", "right"],
        default="auto",
        help="auto uses right half for side-by-side wide videos.",
    )
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--infrared",
        action="store_true",
        help="Treat input as infrared (convert to grayscale, replicate to 3 channels).",
    )
    # Multi-person tracking options
    parser.add_argument(
        "--multi-person",
        action="store_true",
        help="Enable multi-person tracking mode using model.track().",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config for multi-person mode.",
    )
    parser.add_argument(
        "--stale-threshold",
        type=int,
        default=30,
        help="Frames before a missing track_id is cleaned up.",
    )
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


def _build_fall_config(args: argparse.Namespace) -> FallRuleConfig:
    return FallRuleConfig(
        temporal_window=args.temporal_window,
        temporal_min_fall_votes=args.temporal_votes,
        fall_score_threshold=args.fall_score_threshold,
        bbox_aspect_threshold=args.bbox_aspect_threshold,
        torso_horizontal_angle_threshold=args.torso_angle_threshold,
        shoulder_hip_gap_threshold=args.shoulder_hip_gap_threshold,
    )


def _build_state_config(args: argparse.Namespace) -> TemporalStateConfig:
    return TemporalStateConfig(
        window=args.temporal_window,
        confirm_votes=args.temporal_votes,
    )


def _round_or_none(value: float | None, ndigits: int = 4) -> float | None:
    return round(value, ndigits) if value is not None else None


def _temporal_risk_score(row: dict) -> float:
    """Video-level risk score for ROC-AUC ranking.

    ``max_score`` is useful for debugging single-frame pose geometry, but it is
    too sensitive to isolated ADL poses. This score keeps confirmed temporal
    alerts high and damps videos where the state machine never confirms a fall.
    """
    max_score = float(row.get("max_score") or 0.0)
    frames = max(1, int(row.get("frames") or 0))
    person_frames = max(1, int(row.get("person_frames") or 0))
    temporal_fall_frames = int(row.get("temporal_fall_frames") or 0)
    first_alert_frame = row.get("first_alert_frame")

    if temporal_fall_frames <= 0:
        return round(max_score * 0.25, 3)

    confirmed_ratio = temporal_fall_frames / person_frames
    persistence_factor = 0.65 + 0.35 * min(1.0, confirmed_ratio / 0.05)

    if first_alert_frame is None:
        onset_factor = 0.4
    else:
        onset_ratio = max(0.0, min(1.0, float(first_alert_frame) / frames))
        onset_factor = 0.4 + 0.6 * min(1.0, onset_ratio / 0.2)

    return round(max_score * persistence_factor * onset_factor, 3)


def _known_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if isinstance(row.get("expected_fall"), bool)
        and row.get("outcome") != "ERROR"
    ]


def _roc_auc(rows: list[dict], score_key: str) -> float | None:
    known = _known_rows(rows)
    positives = [
        float(row[score_key])
        for row in known
        if row["expected_fall"] is True
    ]
    negatives = [
        float(row[score_key])
        for row in known
        if row["expected_fall"] is False
    ]
    if not positives or not negatives:
        return None

    pair_score = 0.0
    for pos_score in positives:
        for neg_score in negatives:
            if pos_score > neg_score:
                pair_score += 1.0
            elif pos_score == neg_score:
                pair_score += 0.5
    return round(pair_score / (len(positives) * len(negatives)), 4)


def _classification_summary(rows: list[dict], score_key: str, threshold: float) -> dict:
    known = _known_rows(rows)
    tp = fp = tn = fn = 0
    for row in known:
        expected = bool(row["expected_fall"])
        predicted = float(row[score_key]) >= threshold
        if expected and predicted:
            tp += 1
        elif expected and not predicted:
            fn += 1
        elif not expected and predicted:
            fp += 1
        else:
            tn += 1

    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )

    return {
        "threshold": round(threshold, 4),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": round((tp + tn) / total, 4) if total else None,
        "precision": _round_or_none(precision),
        "recall": _round_or_none(recall),
        "specificity": _round_or_none(specificity),
        "f1": _round_or_none(f1),
        "balanced_accuracy": _round_or_none(balanced_accuracy),
    }


def _best_threshold(rows: list[dict], score_key: str) -> dict | None:
    known = _known_rows(rows)
    if not known:
        return None

    scores = sorted({float(row[score_key]) for row in known})
    thresholds = [0.0]
    thresholds.extend((left + right) / 2 for left, right in zip(scores, scores[1:]))
    thresholds.append(1.0)

    candidates = [
        _classification_summary(rows, score_key, threshold)
        for threshold in thresholds
    ]
    return max(
        candidates,
        key=lambda item: (
            item["balanced_accuracy"] or -1.0,
            item["f1"] or -1.0,
            item["accuracy"] or -1.0,
            item["threshold"],
        ),
    )


# ---------------------------------------------------------------------------
# Single-person evaluation
# ---------------------------------------------------------------------------


def process_frame_for_eval(
    result,
    h,
    w,
    roi,
    detector,
    state_machine,
    args,
    max_score: float,
    raw_fall_frames: int,
    temporal_fall_frames: int,
    first_alert_frame: int | None,
    frame_count: int,
) -> tuple[int, int, int, int | None, float]:
    if result.boxes is None or result.keypoints is None:
        detector.reset()
        state_machine.reset()
        return 0, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    selected = select_primary_pose(boxes, keypoints, w, h, roi)

    if selected is None:
        detector.reset()
        state_machine.reset()
        return 0, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score

    person_frames = 1
    box, kpts = selected
    decision = detector.classify(kpts, box)
    if args.decision_mode == "state_machine":
        state_decision = state_machine.update(decision, box, h)
        decision = replace(
            decision,
            temporal_is_fall=state_decision.is_fall_confirmed,
            reason=f"{decision.reason}|{state_decision.state.value}",
        )
    max_score = max(max_score, decision.score)
    if decision.is_fall:
        raw_fall_frames += 1
    if decision.temporal_is_fall:
        temporal_fall_frames += 1
        if first_alert_frame is None:
            first_alert_frame = frame_count

    return person_frames, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score


# ---------------------------------------------------------------------------
# Multi-person evaluation
# ---------------------------------------------------------------------------


def _summarize_multi_frame_metrics(
    person_results,
    raw_fall_frames: int,
    temporal_fall_frames: int,
    first_alert_frame: int | None,
    max_score: float,
    frame_count: int,
) -> tuple[int, int, int, int | None, float]:
    """Update video metrics once per frame, regardless of person count.

    This keeps ``person_frames``, ``raw_fall_frames`` and
    ``temporal_fall_frames`` comparable between single-person and
    multi-person evaluation. A frame containing two falling people still
    counts as one fall frame.
    """
    if not person_results:
        return 0, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score

    max_score = max(max_score, *(pr.decision.score for pr in person_results))
    frame_has_raw_fall = any(pr.decision.is_fall for pr in person_results)
    frame_has_temporal_fall = any(
        pr.decision.temporal_is_fall for pr in person_results
    )

    raw_fall_frames += int(frame_has_raw_fall)
    temporal_fall_frames += int(frame_has_temporal_fall)
    if frame_has_temporal_fall and first_alert_frame is None:
        first_alert_frame = frame_count

    return 1, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score


def process_frame_for_eval_multi(
    result,
    h,
    w,
    tracker_mgr: TrackerManager,
    args,
    max_score: float,
    raw_fall_frames: int,
    temporal_fall_frames: int,
    first_alert_frame: int | None,
    frame_count: int,
) -> tuple[int, int, int, int | None, float]:
    """Process one frame for multi-person evaluation.

    "Any person confirmed fall" counts as a fall for the video.
    """
    tracker_mgr.set_frame(frame_count)

    if result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        tracker_mgr.cleanup_stale()
        return 0, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score

    boxes = result.boxes.xyxy.cpu().numpy()
    keypoints = result.keypoints.data.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()

    track_ids = result.boxes.id
    if track_ids is not None:
        track_ids_np = track_ids.cpu().numpy().astype(int)
    else:
        track_ids_np = None

    if track_ids_np is None:
        tracker_mgr.cleanup_stale()
        return 0, raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score

    person_results = []
    for i in range(len(boxes)):
        tid = int(track_ids_np[i])
        box = boxes[i]
        kpts = keypoints[i]
        det_conf = float(confs[i])

        pr = tracker_mgr.update(tid, kpts, box, h, det_conf)
        person_results.append(pr)

    tracker_mgr.cleanup_stale()
    return _summarize_multi_frame_metrics(
        person_results,
        raw_fall_frames,
        temporal_fall_frames,
        first_alert_frame,
        max_score,
        frame_count,
    )


# ---------------------------------------------------------------------------
# Video evaluation
# ---------------------------------------------------------------------------


def evaluate_video(
    model,
    video_path: Path,
    args: argparse.Namespace,
) -> dict:
    frame_count = 0
    person_frames = 0
    raw_fall_frames = 0
    temporal_fall_frames = 0
    first_alert_frame: int | None = None
    max_score = 0.0

    if args.multi_person:
        tracker_mgr = TrackerManager(
            fall_config=_build_fall_config(args),
            state_config=_build_state_config(args),
            stale_threshold=args.stale_threshold,
            use_state_machine=(args.decision_mode == "state_machine"),
        )

        if args.infrared:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    "video": str(video_path),
                    "expected_fall": expected_from_name(video_path),
                    "predicted_fall": None,
                    "outcome": "ERROR",
                    "frames": 0,
                    "person_frames": 0,
                    "raw_fall_frames": 0,
                    "temporal_fall_frames": 0,
                    "first_alert_frame": None,
                    "max_score": 0.0,
                    "error": "Failed to open video",
                }

            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if args.max_frames > 0 and frame_count > args.max_frames:
                    break

                infrared_frame = to_infrared(frame)
                result = model.track(
                    infrared_frame,
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    tracker=args.tracker,
                    persist=True,
                    verbose=False,
                )[0]
                pf, rf, tf, fa, ms = process_frame_for_eval_multi(
                    result, h, w, tracker_mgr, args, max_score, raw_fall_frames, temporal_fall_frames, first_alert_frame, frame_count
                )
                person_frames += pf
                raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score = rf, tf, fa, ms

            cap.release()
        else:
            for result in model.track(
                source=str(video_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=args.device,
                tracker=args.tracker,
                persist=True,
                stream=True,
                verbose=False,
            ):
                frame_count += 1
                if args.max_frames > 0 and frame_count > args.max_frames:
                    break

                h, w = result.orig_img.shape[:2]
                pf, rf, tf, fa, ms = process_frame_for_eval_multi(
                    result, h, w, tracker_mgr, args, max_score, raw_fall_frames, temporal_fall_frames, first_alert_frame, frame_count
                )
                person_frames += pf
                raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score = rf, tf, fa, ms
    else:
        # Single-person mode
        detector = FallDetector(_build_fall_config(args))
        state_machine = TemporalFallStateMachine(_build_state_config(args))

        if args.infrared:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return {
                    "video": str(video_path),
                    "expected_fall": expected_from_name(video_path),
                    "predicted_fall": None,
                    "outcome": "ERROR",
                    "frames": 0,
                    "person_frames": 0,
                    "raw_fall_frames": 0,
                    "temporal_fall_frames": 0,
                    "first_alert_frame": None,
                    "max_score": 0.0,
                    "error": "Failed to open video",
                }

            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                if args.max_frames > 0 and frame_count > args.max_frames:
                    break

                infrared_frame = to_infrared(frame)
                result = model(infrared_frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
                pf, rf, tf, fa, ms = process_frame_for_eval(
                    result, h, w, args.roi, detector, state_machine, args, max_score, raw_fall_frames, temporal_fall_frames, first_alert_frame, frame_count
                )
                person_frames += pf
                raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score = rf, tf, fa, ms

            cap.release()
        else:
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

                h, w = result.orig_img.shape[:2]
                pf, rf, tf, fa, ms = process_frame_for_eval(
                    result, h, w, args.roi, detector, state_machine, args, max_score, raw_fall_frames, temporal_fall_frames, first_alert_frame, frame_count
                )
                person_frames += pf
                raw_fall_frames, temporal_fall_frames, first_alert_frame, max_score = rf, tf, fa, ms

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

    row = {
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
    row["temporal_risk_score"] = _temporal_risk_score(row)
    return row


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    # ONNX files do not always preserve enough metadata for Ultralytics to infer
    # the task. This project only accepts pose models, so set it explicitly.
    model = YOLO(args.model, task="pose")
    videos = iter_videos(Path(args.source))
    if not videos:
        raise SystemExit(f"no videos found in {args.source}")

    rows = [evaluate_video(model, video, args) for video in videos]
    summary = {
        key: sum(row["outcome"] == key for row in rows)
        for key in ("TP", "FP", "TN", "FN", "UNKNOWN", "ERROR")
    }
    total_known = summary["TP"] + summary["FP"] + summary["TN"] + summary["FN"]
    summary["accuracy"] = (
        round((summary["TP"] + summary["TN"]) / total_known, 4)
        if total_known
        else None
    )
    summary["auc_score_key"] = "temporal_risk_score"
    summary["auc"] = _roc_auc(rows, "temporal_risk_score")
    summary["max_score_auc"] = _roc_auc(rows, "max_score")
    summary["best_threshold"] = _best_threshold(rows, "temporal_risk_score")

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
