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

所有命令均在项目根目录执行：

```powershell
cd D:\huawei\pure-visual-fall-detection-model
if (!(Test-Path .\.venv\Scripts\python.exe)) { python -m venv .venv }
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 评委现场操作命令

以下命令使用仓库内已有样例视频 `datasets\fall_pose\videos\raw\urfall_fall_01_cam0.mp4` 和版本化模型文件，适合现场快速演示。

### 1. 检查环境

```powershell
python tools\compliance\check_environment.py `
  --output reports\environment\runtime_environment.json
```

### 2. 运行单视频可视化推理

```powershell
python scripts\runtime\infer_video.py `
  --source datasets\fall_pose\videos\raw\urfall_fall_01_cam0.mp4 `
  --model artifacts\pytorch\fall_pose_384_best.pt `
  --output runs\judge\infer_urfall_fall_01.mp4 `
  --decision-mode state_machine `
  --roi auto
```

输出：`runs\judge\infer_urfall_fall_01.mp4`，视频中包含人体框、骨架、跌倒分数和状态机结果。

### 3. 运行端到端告警系统

```powershell
python scripts\runtime\run_system.py `
  --source datasets\fall_pose\videos\raw\urfall_fall_01_cam0.mp4 `
  --model artifacts\pytorch\fall_pose_384_best.pt `
  --output-dir runs\judge\system `
  --run-name urfall_fall_01 `
  --decision-mode state_machine `
  --no-popup
```

输出：`runs\judge\system\urfall_fall_01\`，包含 `live.mp4`、`events.jsonl`、`frames.jsonl`、`run_summary.json`、`report.html`，以及告警截图和回放片段。

### 4. 摄像头实时演示

```powershell
python scripts\runtime\run_system.py `
  --source 0 `
  --model artifacts\pytorch\fall_pose_384_best.pt `
  --output-dir runs\judge\camera `
  --no-popup
```

### 5. 批量评估样例视频

```powershell
python scripts\evaluation\evaluate_videos.py `
  --source datasets\fall_pose\videos\raw `
  --model artifacts\pytorch\fall_pose_384_best.pt `
  --output reports\evaluation\judge_video_evaluation.json `
  --csv-output reports\evaluation\judge_video_evaluation.csv `
  --multi-person
```

输出：`reports\evaluation\judge_video_evaluation.json` 和 `reports\evaluation\judge_video_evaluation.csv`，包含 TP、FP、TN、FN、accuracy、AUC、最佳阈值、首个告警帧、告警帧数和最大分数。

### 6. 查看模型部署证据

如需现场重新测速，先安装 ONNX benchmark 依赖：

```powershell
python -m pip install -r requirements-benchmark.txt
```

```powershell
python tools\model\inspect_onnx.py artifacts\onnx\fall_pose_384_fp32.onnx `
  --output reports\model\judge_fp32_model_audit.json

python tools\model\benchmark_onnx.py artifacts\onnx\fall_pose_384_fp32.onnx `
  --input-npy samples\reference_input_384.npy `
  --runs 30 `
  --warmup 5 `
  --output reports\model\judge_fp32_cpu_benchmark.json

python tools\model\analyze_npu_feasibility.py reports\model\judge_fp32_model_audit.json `
  --benchmark reports\model\judge_fp32_cpu_benchmark.json `
  --output reports\model\judge_fp32_npu_feasibility.json
```

### 7. 运行自动化测试

```powershell
python -m pip install pytest
python -m pytest
```

当前测试集覆盖核心规则、状态机、多人跟踪、可视化、ONNX 量化工具、报告生成和评估指标。

## 批量评估

```powershell
python scripts\evaluation\evaluate_videos.py `
  --source datasets\fall_pose\videos\raw `
  --model artifacts\pytorch\fall_pose_384_best.pt `
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
