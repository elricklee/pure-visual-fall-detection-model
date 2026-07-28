# Pure Visual Fall Detection Model

这是一个纯视觉跌倒检测演示项目：用 YOLO pose 提取人体关键点，再用规则评分、时序状态机和多人 track_id 隔离生成跌倒告警。

当前项目不只包含算法模块，也已经提供可视化视频、事件截图、回放片段、JSONL 事件流和 HTML 运行报告。

## 一键跑出演示结果

准备环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

运行多人跌倒告警系统：

```powershell
python scripts\run_system.py `
  --source path\to\demo.mp4 `
  --model yolov8n-pose.pt `
  --output-dir runs\system `
  --no-popup
```

摄像头输入可以把 `--source` 改成 `0`：

```powershell
python scripts\run_system.py --source 0 --model yolov8n-pose.pt --no-popup
```

运行结束后会生成：

- `runs/system/<run_name>/report.html`：可直接打开的成果报告
- `runs/system/<run_name>/live.mp4`：带骨架、人员 ID 和告警状态的视频
- `runs/system/<run_name>/events.jsonl`：每次跌倒告警事件流
- `runs/system/<run_name>/frames.jsonl`：每帧每人的判定 trace，方便测试回看
- `runs/system/<run_name>/events/<event_id>/snapshot.jpg`：告警截图
- `runs/system/<run_name>/events/<event_id>/replay.mp4`：告警前后回放片段
- `runs/system/<run_name>/run_summary.json`：本次运行参数、帧数、FPS、事件数

## HTML 报告

如果已经有一次 `run_system.py` 输出，也可以单独重新生成报告：

```powershell
python scripts\generate_run_report.py --run-dir runs\system\<run_name>
```

报告页会展示：

- 本次运行的事件数、帧数、FPS、分辨率和耗时
- 标注后完整视频
- 每个告警事件的 track_id、状态、置信分、触发帧、触发时间和规则原因
- 每帧每个人的 det、score、state、reason 和 active track_id
- 每个事件的截图和回放视频入口

## 单视频可视化推理

只想快速生成一个带框视频时，用：

```powershell
python scripts\infer_video.py `
  --source path\to\demo.mp4 `
  --model yolov8n-pose.pt `
  --output runs\infer\fall_demo.mp4 `
  --multi-person
```

这条路径只输出标注视频，不生成事件中心和 HTML 报告。

## 批量评估

对一批视频输出 JSON/CSV 指标：

```powershell
python scripts\evaluate_videos.py `
  --source datasets\fall_pose\videos\raw `
  --model yolov8n-pose.pt `
  --output reports\video_eval.json `
  --csv-output reports\video_eval.csv `
  --multi-person
```

评估报告包含 TP、FP、TN、FN、accuracy、每个视频的首个告警帧、最大分数和告警帧数。

## 核心模块

- `fall_detection/fall_logic.py`：单帧规则评分和投票
- `fall_detection/state_machine.py`：跌倒确认、恢复和误报抑制状态机
- `fall_detection/tracker_manager.py`：按 track_id 隔离每个人的检测器和状态机
- `fall_detection/visualization.py`：骨架、bbox、状态横幅和多人状态绘制
- `scripts/run_system.py`：面向演示的事件中心入口
- `scripts/generate_run_report.py`：运行结果 HTML 报告生成

## 模型与部署工具

仓库里还保留了 ONNX 导出、量化、基准测试和 NPU 可行性分析工具：

- `scripts/export_onnx.py`
- `scripts/quantize_onnx.py`
- `tools/inspect_onnx.py`
- `tools/benchmark_onnx.py`
- `tools/analyze_npu_feasibility.py`
- `tools/check_compliance.py`

这些工具输出到 `reports/`，用于性能、模型审计和比赛材料整理。
