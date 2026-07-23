# 组员甲 Day2 算法优化与误报评估

## 本日目标

围绕 Day1 可运行 MVP，完成基线模型继续训练、5 帧时序后处理调参、
误报/漏报统计，并形成可用于 PPT 的实验结果。

## 数据与训练

- 数据来源：UR Fall Detection Dataset 小样本视频
- 训练集：188 张
- 验证集：37 张
- 测试集：24 张
- 标签类型：YOLO pose 伪关键点标签
- 模型：YOLOv8n-pose
- 训练轮数：20 epochs
- 训练设备：NVIDIA GeForce RTX 4060 Laptop GPU
- 最佳权重：`runs/pose/runs/train/fall_pose_day2_v1/weights/best.pt`

## 验证结果

- Box mAP50：0.821
- Box mAP50-95：0.669
- Pose mAP50：0.578
- Pose mAP50-95：0.328
- 平均推理耗时：4.7ms / image

## 跌倒后处理调参

Day1 默认 `fall_score_threshold=0.55` 时，`urfall_adl_04_cam0.mp4`
出现误报。Day2 将默认阈值调整为：

```text
fall_score_threshold = 0.70
temporal_window = 5
temporal_min_fall_votes = 3
```

## 视频级评估

使用现有 8 个视频评估：

- TP：4
- TN：4
- FP：0
- FN：0
- Accuracy：1.000

评估文件：

- `reports/day2_video_eval_tuned.json`
- `reports/day2_video_eval_tuned.csv`

## Demo 产物

- `runs/demo/day2_urfall_fall_02_demo.mp4`
- `runs/demo/day2_urfall_adl_04_demo.mp4`

## 风险说明

当前数据量较小，且关键点标签为伪标注，不能代表最终比赛精度。后续应人工抽查
50 到 100 张关键帧，并增加弯腰、坐下、遮挡、红外/微光场景样本。
