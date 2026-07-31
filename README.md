# Pure Visual Fall Detection Model

纯视觉跌倒检测 MVP。主线是：YOLOv8 pose 提取人体关键点，几何规则给出跌倒风险分数，时序状态机确认事件，多人模式按 `track_id` 独立维护状态。

## 当前主线

- 实时/视频告警入口：`scripts/runtime/run_system.py`
- 单视频标注入口：`scripts/runtime/infer_video.py`
- 批量评估入口：`scripts/evaluation/evaluate_videos.py`
- 核心算法包：`fall_detection/`
- 当前模型产物：`artifacts/pytorch/yolov8n-pose.pt`、`artifacts/pytorch/fall_pose_384_best.pt`、`artifacts/onnx/fall_pose_384_int8_qdq.onnx`
- 当前证据目录：`reports/model/`、`reports/evaluation/`、`reports/compliance/`

## 项目结构

- `fall_detection/`：跌倒检测核心包。
- `scripts/data/`：数据准备、标注、增强和可视化。
- `scripts/model/`：训练、ONNX 导出和 INT8 量化。
- `scripts/runtime/`：视频推理、端到端告警和运行报告。
- `scripts/evaluation/`：视频评估、ONNX 评估和报告对比。
- `tools/model/`：模型审计、CPU 测速和 NPU 可行性分析。
- `tools/compliance/`：环境检查和赛题硬指标检查。
- `tools/reports/`：报告、PPT 和交付物生成。
- `scripts/*.py`、`tools/*.py`：常用旧路径的兼容入口。
- `configs/`：数据集、模型导出和合规配置。
- `artifacts/`：版本化模型产物，只保留当前需要复现的权重和 ONNX。
- `samples/`：ONNX 对齐和工具测试使用的固定样例。
- `reports/`：当前评估、审计、合规证据；历史材料统一在 `reports/archive/`。
- `docs/`：部署、数据生成、提交检查等说明；历史文档统一在 `docs/archive/`。
- `deliverables/`：提交材料和 PPT 产物。
- `datasets/`、`runs/`、`.venv/`、缓存目录：本地生成或环境内容，不进入版本化主线。

## 快速运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

运行端到端告警系统：

```powershell
python scripts\runtime\run_system.py `
  --source path\to\demo.mp4 `
  --model artifacts\pytorch\yolov8n-pose.pt `
  --output-dir runs\system `
  --no-popup
```

摄像头输入：

```powershell
python scripts\runtime\run_system.py --source 0 --model artifacts\pytorch\yolov8n-pose.pt --no-popup
```

输出目录会生成标注视频、事件流、帧级 trace、告警截图、回放片段、运行摘要和 HTML 报告。

## 批量评估

```powershell
python scripts\evaluation\evaluate_videos.py `
  --source datasets\fall_pose\videos\raw `
  --model artifacts\pytorch\yolov8n-pose.pt `
  --output reports\evaluation\video_evaluation.json `
  --csv-output reports\evaluation\video_evaluation.csv `
  --multi-person
```

评估输出包含 TP、FP、TN、FN、accuracy、AUC、最佳阈值、首个告警帧、告警帧数和最大分数。默认 AUC 使用 `temporal_risk_score`，它结合状态机确认、告警持续性和告警时机；`max_score` 只保留为单帧姿态峰值，容易受 ADL 瞬时姿态尖峰影响。

## 核心模块

- `fall_detection/fall_logic.py`：单帧几何评分。
- `fall_detection/state_machine.py`：跌倒确认、恢复和误报抑制状态机。
- `fall_detection/tracker_manager.py`：多人 `track_id` 隔离。
- `fall_detection/visualization.py`：骨架、bbox、状态横幅和多人状态绘制。
- `scripts/runtime/generate_run_report.py`：HTML 运行报告生成。

## 部署与验证

- ONNX 导出：`scripts/model/export_onnx.py`
- INT8 量化：`scripts/model/quantize_onnx.py`
- ONNX 审计：`tools/model/inspect_onnx.py`
- CPU 参考测速：`tools/model/benchmark_onnx.py`
- NPU 可行性分析：`tools/model/analyze_npu_feasibility.py`
- 赛题硬指标检查：`tools/compliance/check_compliance.py`

## 仓库收敛口径

当前有效材料保留在 `reports/model/`、`reports/evaluation/`、`reports/compliance/` 和 `reports/summaries/`。

重要限制：NPU 延迟和运行内存仍需真实板卡实测。
