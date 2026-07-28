from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fall_detection import FallRuleConfig, TemporalStateConfig
from fall_detection.image_utils import to_infrared
from fall_detection.tracker_manager import TrackerManager
from fall_detection.visualization import draw_multi_decision, draw_multi_status_banner, draw_pose
from scripts.generate_run_report import generate_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-person fall alert system and export artifacts.")
    parser.add_argument("--source", required=True, help="Video file, webcam id, or stream URL.")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--output-dir", default="runs/system", help="Root output directory.")
    parser.add_argument("--run-name", default=None, help="Subfolder name under output-dir (default: timestamp).")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--stale-threshold", type=int, default=30)
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--temporal-votes", type=int, default=3)
    parser.add_argument("--fall-score-threshold", type=float, default=0.70)
    parser.add_argument("--bbox-aspect-threshold", type=float, default=1.25)
    parser.add_argument("--torso-angle-threshold", type=float, default=35.0)
    parser.add_argument("--shoulder-hip-gap-threshold", type=float, default=0.22)
    parser.add_argument("--decision-mode", choices=["state_machine", "vote"], default="state_machine")
    parser.add_argument("--infrared", action="store_true")
    parser.add_argument("--pre-seconds", type=float, default=10.0)
    parser.add_argument("--post-seconds", type=float, default=5.0)
    parser.add_argument("--cooldown-seconds", type=float, default=10.0)
    parser.add_argument("--webhook", default=None, help="HTTP endpoint for POST JSON alerts.")
    parser.add_argument("--no-sound", action="store_true")
    parser.add_argument("--no-popup", action="store_true")
    parser.add_argument("--no-report", action="store_true", help="Skip HTML report generation.")
    return parser.parse_args()


def _run_name(default: str | None = None) -> str:
    if default:
        return default
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _write_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _post_webhook(url: str, payload: dict) -> None:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            resp.read()
    except urllib.error.URLError:
        return


def _post_webhook_async(url: str, payload: dict) -> threading.Thread:
    """Send a webhook without blocking the video inference loop."""
    thread = threading.Thread(target=_post_webhook, args=(url, payload), daemon=True)
    thread.start()
    return thread


def _should_emit_fall_event(
    track_id: int,
    is_confirmed: bool,
    now_sec: float,
    last_event_t: dict[int, float],
    active_alert_ids: set[int],
    cooldown_seconds: float,
) -> bool:
    """Return True only when a track enters a new confirmed-fall incident."""
    track_id = int(track_id)
    if not is_confirmed:
        active_alert_ids.discard(track_id)
        return False

    if track_id in active_alert_ids:
        return False

    active_alert_ids.add(track_id)
    previous = float(last_event_t.get(track_id, -1e18))
    return now_sec - previous >= max(0.0, float(cooldown_seconds))


def _play_sound() -> None:
    try:
        import winsound

        winsound.MessageBeep(winsound.MB_ICONHAND)
    except Exception:
        return


def _show_popup(title: str, message: str) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning(title, message)
        root.destroy()
    except Exception:
        return


def _show_popup_async(title: str, message: str) -> None:
    thread = threading.Thread(target=_show_popup, args=(title, message), daemon=True)
    thread.start()


def _write_video(path: Path, frames: list, fps: float, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()


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


def _process_frame_multi(
    model,
    tracker_mgr: TrackerManager,
    frame,
    display_frame,
    args: argparse.Namespace,
    frame_height: int,
    frame_index: int,
    fps: float,
    trace_path: Path,
    writer,
) -> list:
    tracker_mgr.set_frame(frame_index)
    result = model.track(
        frame,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        tracker=args.tracker,
        persist=True,
        verbose=False,
    )[0]

    person_results = []
    detected_person_count = 0
    if result.boxes is not None and result.keypoints is not None and len(result.boxes) > 0:
        detected_person_count = len(result.boxes)
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        track_ids = result.boxes.id
        if track_ids is not None:
            track_ids_np = track_ids.cpu().numpy().astype(int)
        else:
            track_ids_np = None

        if track_ids_np is not None:
            for i in range(len(boxes)):
                tid = int(track_ids_np[i])
                box = boxes[i]
                kpts = keypoints[i]
                det_conf = float(confs[i])

                pr = tracker_mgr.update(tid, kpts, box, frame_height, det_conf)
                person_results.append(pr)

                draw_pose(display_frame, kpts)
                draw_multi_decision(display_frame, box, pr.decision, tid, det_conf)

    tracker_mgr.cleanup_stale()
    frame_trace = {
        "type": "frame_trace",
        "frame_index": frame_index,
        "t_sec": round(frame_index / max(1e-6, fps), 3),
        "active_ids": tracker_mgr.active_ids,
        "person_count": len(person_results),
        "detected_person_count": detected_person_count,
        "untracked_detections": max(0, detected_person_count - len(person_results)),
        "detections": [
            {
                "track_id": int(pr.track_id),
                "det_conf": float(pr.det_conf),
                "score": float(pr.decision.score),
                "is_fall": bool(pr.decision.is_fall),
                "temporal_is_fall": bool(pr.decision.temporal_is_fall),
                "state": pr.state_decision.state.value if pr.state_decision else "VOTE",
                "reason": pr.decision.reason,
                "box_xyxy": [float(v) for v in pr.box_xyxy],
            }
            for pr in person_results
        ],
    }
    _write_jsonl(trace_path, frame_trace)
    draw_multi_status_banner(display_frame, person_results)
    writer.write(display_frame)
    return person_results


def main() -> None:
    args = parse_args()
    run_name = _run_name(args.run_name)
    run_dir = Path(args.output_dir) / run_name
    events_dir = run_dir / "events"
    events_jsonl = run_dir / "events.jsonl"
    trace_jsonl = run_dir / "frames.jsonl"
    out_video = run_dir / "live.mp4"
    summary_path = run_dir / "run_summary.json"
    report_path = run_dir / "report.html"

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. Run: pip install -r requirements.txt") from exc

    model = YOLO(args.model, task="pose")

    cap_source: str | int = str(args.source)
    if isinstance(cap_source, str) and cap_source.isdigit() and not Path(cap_source).exists():
        cap_source = int(cap_source)
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (width, height)

    out_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps, size)
    if not writer.isOpened():
        cap.release()
        raise SystemExit(f"Failed to open output video writer: {out_video}")

    pre_frames = max(0, int(round(args.pre_seconds * fps)))
    post_frames = max(0, int(round(args.post_seconds * fps)))

    pre_buffer = []
    pending = []
    last_event_t = {}
    active_alert_ids: set[int] = set()
    webhook_threads: list[threading.Thread] = []

    tracker_mgr = TrackerManager(
        fall_config=_build_fall_config(args),
        state_config=_build_state_config(args),
        stale_threshold=args.stale_threshold,
        use_state_machine=(args.decision_mode == "state_machine"),
    )

    start_wall = time.time()
    frame_index = -1
    event_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        raw_frame = to_infrared(frame) if args.infrared else frame
        display_frame = raw_frame.copy()

        person_results = _process_frame_multi(
            model=model,
            tracker_mgr=tracker_mgr,
            frame=raw_frame,
            display_frame=display_frame,
            args=args,
            frame_height=height,
            frame_index=frame_index,
            fps=fps,
            trace_path=trace_jsonl,
            writer=writer,
        )

        for clip in list(pending):
            clip["frames"].append(display_frame.copy())
            clip["remaining"] -= 1
            if clip["remaining"] <= 0:
                _write_video(clip["clip_path"], clip["frames"], fps, size)
                _write_jsonl(
                    events_jsonl,
                    {
                        "type": "event_clip_saved",
                        "event_id": clip["event_id"],
                        "clip_path": str(clip["clip_path"]),
                        "frame_index_end": frame_index,
                    },
                )
                pending.remove(clip)

        now_sec = frame_index / max(1e-6, fps)
        active_alert_ids.intersection_update(int(tid) for tid in tracker_mgr.active_ids)
        for pr in person_results:
            tid = int(pr.track_id)
            if not _should_emit_fall_event(
                track_id=tid,
                is_confirmed=bool(pr.decision.temporal_is_fall),
                now_sec=now_sec,
                last_event_t=last_event_t,
                active_alert_ids=active_alert_ids,
                cooldown_seconds=args.cooldown_seconds,
            ):
                continue

            event_count += 1
            event_id = f"{run_name}_p{tid}_{frame_index:06d}"
            event_dir = events_dir / event_id
            event_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = event_dir / "snapshot.jpg"
            clip_path = event_dir / "replay.mp4"

            cv2.imwrite(str(snapshot_path), display_frame)

            state = pr.state_decision.state.value if pr.state_decision else "VOTE"
            alert_payload = {
                "type": "fall_alert",
                "event_id": event_id,
                "track_id": tid,
                "state": state,
                "score": float(pr.decision.score),
                "reason": pr.decision.reason,
                "frame_index": frame_index,
                "t_sec": round(now_sec, 3),
                "snapshot_path": str(snapshot_path),
                "clip_path": str(clip_path),
                "run_dir": str(run_dir),
            }
            _write_jsonl(
                events_jsonl,
                {
                    "type": "event_start",
                    "alert": alert_payload,
                },
            )

            if not args.no_sound:
                _play_sound()
            if not args.no_popup:
                _show_popup_async("Fall Alert", f"event={event_id} person=#{tid} score={pr.decision.score}")
            if args.webhook:
                webhook_threads.append(_post_webhook_async(args.webhook, alert_payload))

            pre_clip = pre_buffer[-pre_frames:] if pre_frames > 0 else []
            frames = [f.copy() for f in pre_clip] + [display_frame.copy()]
            pending.append(
                {
                    "event_id": event_id,
                    "clip_path": clip_path,
                    "frames": frames,
                    "remaining": post_frames,
                }
            )
            last_event_t[tid] = now_sec

        pre_buffer.append(display_frame.copy())
        if pre_frames > 0 and len(pre_buffer) > pre_frames:
            pre_buffer = pre_buffer[-pre_frames:]

    cap.release()
    writer.release()

    for clip in list(pending):
        _write_video(clip["clip_path"], clip["frames"], fps, size)
        _write_jsonl(
            events_jsonl,
            {
                "type": "event_clip_saved",
                "event_id": clip["event_id"],
                "clip_path": str(clip["clip_path"]),
                "frame_index_end": frame_index,
            },
        )

    for thread in webhook_threads:
        thread.join(timeout=3.5)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "source": str(args.source),
        "model": str(args.model),
        "fps": float(fps),
        "width": width,
        "height": height,
        "frames": int(frame_index + 1),
        "events": int(event_count),
        "event_trigger_policy": "confirmed_state_entry_with_per_track_cooldown",
        "args": vars(args),
        "started_at": datetime.fromtimestamp(start_wall).isoformat(timespec="seconds"),
        "ended_at": datetime.fromtimestamp(time.time()).isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - start_wall, 3),
        "artifacts": {
            "live_video": str(out_video),
            "events_jsonl": str(events_jsonl),
            "frames_jsonl": str(trace_jsonl),
            "events_dir": str(events_dir),
            "report_html": str(report_path),
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.no_report:
        generate_report(run_dir, report_path)
    print(f"saved: {out_video}")
    print(f"saved: {events_jsonl}")
    print(f"saved: {trace_jsonl}")
    print(f"saved: {summary_path}")
    if not args.no_report:
        print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
