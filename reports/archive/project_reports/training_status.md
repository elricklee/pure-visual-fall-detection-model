# 甲模型当前训练状态

更新时间：2026-07-24

## 可核验结论

- 模型：YOLOv8n-pose；
- 任务：单类 `person` 人体姿态检测；
- 基础权重：`yolov8n-pose.pt`；
- 训练目录：`runs/train/fall_pose_v3/`；
- 训练配置：`runs/train/fall_pose_v3/args.yaml`；
- 训练结果：`runs/train/fall_pose_v3/results.csv`；
- 训练记录共 50 个 epoch；
- 训练输入尺寸：640；
- Batch：16；
- 训练设备参数：`device=0`；
- 最佳权重：`runs/train/fall_pose_v3/weights/best.pt`；
- 最后一轮权重：`runs/train/fall_pose_v3/weights/last.pt`。

`best.pt` 已复制为乙的交付权重：

```text
artifacts/pytorch/yolov8n-pose.pt
```

## 训练指标

`results.csv` 最后一轮记录：

| 指标 | 数值 |
|---|---:|
| Box mAP50 | 0.99500 |
| Box mAP50-95 | 0.85912 |
| Pose mAP50 | 0.87227 |
| Pose mAP50-95 | 0.44447 |

训练记录中的最高 Pose mAP50-95 出现在第 1 个 epoch，为 0.78125。小样本和伪关键点标签可能导致指标波动，不能只依据单个最高值判断泛化能力。

## 当前数据状态

当前工作区可见：

| 划分 | 图片 | 标签 |
|---|---:|---:|
| train | 100 | 100 |
| val | 11 | 11 |
| test | 0 | 0 |

这些数量是当前目录状态；训练时的精确样本数量没有被独立写入训练日志，因此不能据此反推训练当时的数据规模。

## 模型语义

模型原生类别只有 `person`，负责输出人体框和 COCO 17 个关键点。

`fall/normal` 不是 YOLO 原生类别，而是以下模块的后处理结果：

- `fall_detection/fall_logic.py`；
- `fall_detection/state_machine.py`；
- 5 帧时序窗口；
- 默认跌倒分数阈值 0.70；
- 默认确认票数 3。

## 当前限制

- 数据量较小；
- 标签包含自动生成或伪关键点标签，仍需人工抽查；
- 当前没有独立测试集；
- 尚未覆盖足量弯腰、坐下、遮挡、微光和红外场景；
- 多人场景当前选择一个主要人体更新状态机，尚未按 `track_id` 为每个人维护独立状态；
- 当前 ONNX 是 Day 2 基线版本，不是最终优化模型；
- 尚无真实 NPU 延迟和 NPU 运行内存数据。

## 与旧报告的关系

`reports/archive/project_reports/member_a_day2_eval.md` 中的 20 epoch、旧权重路径及数据数量与当前工作区产物不一致。本文件以实际存在的 `args.yaml`、`results.csv` 和权重文件为准。
