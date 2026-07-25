from __future__ import annotations

import argparse
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
)
from fall_detection.detection_selection import select_primary_pose_index
from fall_detection.image_utils import to_infrared
from fall_detection.visualization import draw_decision, draw_pose, draw_status_banner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fall detection on a video/image source.")
    parser.add_argument("--source", required=True, help="Video file, image, webcam id, or stream URL.")
    parser.add_argument("--model", default="yolov8n-pose.pt", help="YOLO pose model path.")
    parser.add_argument("--output", default="runs/infer/fall_demo.mp4", help="Output video path.")
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
    parser.add_argument(
        "--infrared",
        action="store_true",
        help="Treat input as infrared (convert to grayscale, replicate to 3 channels).",
    )
    return parser.parse_args()


def process_frame(
    frame,
    h,
    w,
    model,
    detector,
    state_machine,
    args,
    display_frame,
    writer,
) -> tuple[str, bool, float]:
    if args.infrared:
        inference_frame = to_infrared(frame)
    else:
        inference_frame = frame

    result = model(inference_frame, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]

    state_label = "NO_PERSON"
    alert = False
    score = 0.0
    det_conf = 0.0
    if result.boxes is not None and result.keypoints is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        keypoints = result.keypoints.data.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        selected_idx = select_primary_pose_index(boxes, keypoints, w, h, args.roi)

        if selected_idx is not None:
            box = boxes[selected_idx]
            kpts = keypoints[selected_idx]
            det_conf = float(confs[selected_idx])
            decision = detector.classify(kpts, box)
            state_label = "VOTE"
            score = decision.score
            if args.decision_mode == "state_machine":
                state_decision = state_machine.update(decision, box, h)
                state_label = state_decision.state.value
                decision = replace(
                    decision,
                    temporal_is_fall=state_decision.is_fall_confirmed,
                    reason=decision.reason,
                )
            alert = decision.temporal_is_fall
            draw_pose(display_frame, kpts)
            draw_decision(display_frame, box, decision, det_conf)
        else:
            detector.reset()
            state_machine.reset()
    else:
        detector.reset()
        state_machine.reset()

    draw_status_banner(display_frame, state_label, alert, score)
    if writer is not None:
        writer.write(display_frame)

    return state_label, alert, score


def main() -> None:
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "ultralytics is not installed. Run: pip install -r requirements.txt"
        ) from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
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
    state_machine = TemporalFallStateMachine(
        TemporalStateConfig(
            window=args.temporal_window,
            confirm_votes=args.temporal_votes,
        )
    )

    writer: cv2.VideoWriter | None = None
    fps = 25.0

    if args.infrared:
        cap = cv2.VideoCapture(str(args.source))
        if not cap.isOpened():
            raise SystemExit(f"Failed to open video source: {args.source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            display_frame = to_infrared(frame)
            process_frame(frame, h, w, model, detector, state_machine, args, display_frame, writer)

        cap.release()
    else:
        for result in model.predict(
            source=args.source,
            imgsz=args.imgsz,
            conf=args.conf,
            device=args.device,
            stream=True,
            verbose=False,
        ):
            frame = result.orig_img.copy()
            h, w = frame.shape[:2]

            if writer is None:
                source_fps = getattr(result, "fps", None)
                if isinstance(source_fps, (int, float)) and source_fps > 0:
                    fps = float(source_fps)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

            process_frame(frame, h, w, model, detector, state_machine, args, frame, writer)

    if writer is not None:
        writer.release()
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()