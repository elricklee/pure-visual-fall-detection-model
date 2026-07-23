# Pure Visual Fall Detection Model

面向低算力端侧设备的纯视觉跌倒检测 MVP。Day1 基线采用 `YOLOv8n-pose`
做人形关键点检测，再用轻量姿态几何规则和短时序平滑输出跌倒告警。

## Day1 目标

组员甲主攻：

1. 搭建 PyTorch + Ultralytics 训练环境。
2. 部署 `YOLOv8n-pose` 基线模型。
3. 完成人体关键点到跌倒姿态判断的基础逻辑。
4. 提供首轮预训练入口和视频推理 Demo。

交叉协助：

1. 用数据集校验脚本检查抽帧和标注结构。
2. 输出 Day1 文档框架，方便 PPT 和答辩材料同步。

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

训练基线模型：

```powershell
python scripts\train_pose_baseline.py --data configs\fall_pose.yaml --epochs 50 --imgsz 640
```

检查数据集目录和 YOLO pose 标注：

```powershell
python scripts\validate_dataset.py --data configs\fall_pose.yaml
```

视频推理：

```powershell
python scripts\infer_video.py --source path\to\demo.mp4 --output runs\demo_fall.mp4
```

更多 Day1 分步说明见 [docs/day1_member_a.md](docs/day1_member_a.md)。
