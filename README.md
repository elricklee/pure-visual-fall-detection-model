# Pure Visual Fall Detection Model

纯视觉跌倒检测 MVP。主线是：YOLOv8 pose 提取人体关键点，几何规则给出跌倒风险分数，时序状态机确认事件，多人模式按 `track_id` 独立维护状态。

## 当前主线

- 实时/视频告警入口：`scripts/run_system.py`
- 单视频标注入口：`scripts/infer_video.py`
- 批量评估入口：`scripts/evaluate_videos.py`
- 核心算法包：`fall_detection/`
- 当前模型产物：`artifacts/pytorch/yolov8n-pose.pt`、`artifacts/onnx/fall_pose_384_int8_qdq.onnx`
- 当前证据目录：`reports/model/`、`reports/evaluation/`、`reports/compliance/`

## 快速运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

运行端到端告警系统：

```powershell
python scripts\run_system.py `
  --source path\to\demo.mp4 `
  --model yolov8n-pose.pt `
  --output-dir runs\system `
  --no-popup
```

摄像头输入：

```powershell
python scripts\run_system.py --source 0 --model yolov8n-pose.pt --no-popup
```

输出目录会生成标注视频、事件流、帧级 trace、告警截图、回放片段、运行摘要和 HTML 报告。

## 批量评估

```powershell
python scripts\evaluate_videos.py `
  --source datasets\fall_pose\videos\raw `
  --model yolov8n-pose.pt `
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
- `scripts/generate_run_report.py`：HTML 运行报告生成。

## 部署与验证

- ONNX 导出：`scripts/export_onnx.py`
- INT8 量化：`scripts/quantize_onnx.py`
- ONNX 审计：`tools/inspect_onnx.py`
- CPU 参考测速：`tools/benchmark_onnx.py`
- NPU 可行性分析：`tools/analyze_npu_feasibility.py`
- 赛题硬指标检查：`tools/check_compliance.py`

## 仓库收敛口径

当前有效材料保留在 `reports/model/`、`reports/evaluation/`、`reports/compliance/` 和 `reports/summaries/`。历史 Day1/Day2 过程材料已移入 `docs/archive/` 和 `reports/archive/`，只作追溯，不作为当前结论入口。

重要限制：当前视频真值样本仍少；`pseudo_ir` 是灰度复制的伪红外流程验证，不等于真实红外传感器数据；NPU 延迟和运行内存仍需真实板卡实测。
