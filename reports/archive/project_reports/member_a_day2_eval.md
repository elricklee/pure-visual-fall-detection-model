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

## 算法升级

在原有 5 帧投票基础上，新增姿态时序状态机：

```text
NORMAL -> SUSPECT_FALL -> FALL_CONFIRMED -> RECOVERY
```

状态机同时参考：

- 单帧跌倒姿态分数
- 连续帧下落趋势
- 躯干角度快速变化
- 多帧跌倒投票确认
- 多帧正常姿态恢复

相比单帧判断，状态机更适合解释“跌倒是一个连续事件”，能作为 PPT
中的算法创新点。

跌倒分数由离散规则升级为连续评分：

```text
fall_score = 0.35 * bbox_score
           + 0.45 * torso_score
           + 0.20 * shoulder_hip_gap_score
```

其中三个子分数均在 0 到 1 之间线性变化。`fall_score` 是可解释的风险分数，
不是 YOLO 检测置信度，也不是经过概率校准的真实概率。

## 视频级评估

使用现有 8 个视频评估：

- TP：4
- TN：4
- FP：0
- FN：0
- Accuracy：1.000

状态机回归评估保持上述结果。`datasets/fall_pose/videos/raw` 中另有
10 个 `20240912_*.mp4` 本地视频，因文件名没有 fall/adl 标签，当前只做
推理记录，不纳入准确率统计。

评估文件：

- `reports/archive/day2_video_eval/day2_video_eval_tuned.json`
- `reports/archive/day2_video_eval/day2_video_eval_tuned.csv`
- `reports/archive/day2_video_eval/day2_video_eval_state_machine.json`
- `reports/archive/day2_video_eval/day2_video_eval_state_machine.csv`
- `reports/archive/day2_video_eval/day2_video_eval_state_machine_fixed.json`
- `reports/archive/day2_video_eval/day2_video_eval_state_machine_fixed.csv`

状态机版 Demo：

- `runs/demo/day2_continuous_state_machine/`

显示修正：

- 自动识别 UR Fall 左右双画面，只选右侧 RGB 主人形
- 左上角独立显示 `STATE: NORMAL / SUSPECT_FALL / FALL_CONFIRMED`
- 检测框显示连续 `fall_score`，疑似姿态标记为 `FALL_POSE`
- 只有状态机确认后才显示红色 `FALL` 报警
- 避免同一帧左右两个人形重复更新状态机

最终连续评分评估文件：

- `reports/archive/day2_video_eval/day2_video_eval_continuous_state.json`
- `reports/archive/day2_video_eval/day2_video_eval_continuous_state.csv`

## Demo 产物

- `runs/demo/day2_urfall_fall_02_demo.mp4`
- `runs/demo/day2_urfall_adl_04_demo.mp4`

## 风险说明

当前数据量较小，且关键点标签为伪标注，不能代表最终比赛精度。后续应人工抽查
50 到 100 张关键帧，并增加弯腰、坐下、遮挡、红外/微光场景样本。
