# Script Layout

`scripts/` keeps thin compatibility entry points for the most-used commands. New code should import or execute the categorized modules below.

## Runtime

- `runtime/run_system.py`: end-to-end alert system, event export, and HTML report generation.
- `runtime/infer_video.py`: annotated single-video or multi-person inference.
- `runtime/generate_run_report.py`: HTML report renderer for a completed run.

## Evaluation

- `evaluation/evaluate_videos.py`: video-level fall alert evaluation.
- `evaluation/evaluate_pose_onnx.py`: ONNX pose output evaluation.
- `evaluation/compare_video_reports.py`: compare two video evaluation JSON reports.
- `evaluation/compare_onnx_outputs.py`: compare PyTorch and ONNX outputs.

## Model

- `model/train_pose_baseline.py`: YOLO pose training entry point.
- `model/export_onnx.py`: export PyTorch weights to ONNX.
- `model/quantize_onnx.py`: ONNX INT8 quantization utilities.

## Data

- `data/scaffold_dataset.py`: create the dataset folder skeleton.
- `data/extract_frames.py`: extract frames from raw videos.
- `data/split_dataset.py`: split images and labels into train/val/test.
- `data/validate_dataset.py`: validate YOLO dataset structure.
- `data/auto_label_pose.py` and `data/make_labels.py`: generate pose labels.
- `data/prepare_infrared_data.py`: prepare pseudo-infrared data.
- `data/enhance_data.py`: apply data enhancement.
- `data/visualize_pose_labels.py`: visualize pose labels.
- `data/create_multi_person_fixture.py`: create a multi-person validation fixture.
