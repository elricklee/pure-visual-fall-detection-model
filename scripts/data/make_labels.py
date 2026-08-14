from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--model", default="artifacts/pytorch/yolov8n-pose.pt")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--class-id", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    from ultralytics import YOLO

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    results = model.predict(
        source=str(image_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        stream=True,
        verbose=False,
    )

    for result in results:
        img_path = Path(result.path)
        h, w = result.orig_shape[:2]

        if result.keypoints is None or result.keypoints.data.shape[0] == 0:
            # 没检测到人 → 跳过，只复制图片
            print(f"[skip] no person: {img_path.name}")
            continue

        kpts = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()       # (N, 4)

        # 选信心最高的人
        confs = kpts[:, :, 2].mean(axis=1)
        best_idx = int(np.argmax(confs))
        best_kpts = kpts[best_idx]
        best_box = boxes[best_idx]

        x1, y1, x2, y2 = best_box
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # 归一化
        cx_n, cy_n = cx / w, cy / h
        bw_n, bh_n = bw / w, bh / h
        kpts_n = best_kpts.copy()
        kpts_n[:, 0] /= w
        kpts_n[:, 1] /= h

        # YOLO pose 格式: class cx cy w h kpt_x kpt_y vis ...(×17)
        line_parts = [args.class_id, cx_n, cy_n, bw_n, bh_n]
        for kx, ky, kv in kpts_n:
            line_parts.extend([kx, ky, kv])

        label_path = label_dir / img_path.with_suffix(".txt").name
        with open(label_path, "w") as f:
            f.write(" ".join(f"{v:.6f}" for v in line_parts))

        print(f"[ok] {img_path.name}")

    print("done")


if __name__ == "__main__":
    main()
