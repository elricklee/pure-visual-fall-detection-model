# Day1 组员甲执行说明

## 你今天要完成什么

Day1 的核心不是追求高精度，而是先跑通一个可复现的算法基线：

1. 环境能装好。
2. YOLOv8n-pose 能训练或加载预训练权重。
3. 视频输入后能检测人体关键点。
4. 根据关键点给出 `fall` / `normal` 判断。
5. 能保存带框、关键点、告警状态的视频结果。

## 分步执行

### 1. 建环境

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果机器有 NVIDIA GPU，请先按本机 CUDA 版本安装对应 PyTorch，再安装
`requirements.txt` 中剩余依赖。

### 2. 准备数据

推荐目录：

```text
datasets/fall_pose/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

每张图对应一个同名 `.txt` 标注文件。YOLO pose 单行格式：

```text
class x_center y_center width height kpt_x kpt_y visible ...
```

坐标均为 0 到 1 归一化值。`visible` 建议使用 COCO 约定：`0` 不可见，
`1` 标注但遮挡，`2` 可见。

### 3. 校验数据

```powershell
python scripts\validate_dataset.py --data configs\fall_pose.yaml
```

重点看三件事：

1. `images` 和 `labels` 是否一一对应。
2. 每行关键点数量是否为 17 个。
3. train / val 是否都有样本。

### 4. 启动首轮训练

```powershell
python scripts\train_pose_baseline.py --data configs\fall_pose.yaml --epochs 50 --imgsz 640
```

如果还没有自己的数据，也可以先跳过训练，直接用 Ultralytics 自动下载的
`yolov8n-pose.pt` 做 Demo 推理。

### 5. 跑视频 Demo

```powershell
python scripts\infer_video.py --source path\to\demo.mp4 --output runs\demo_fall.mp4
```

输出视频中：

- 红色框：判定为跌倒。
- 绿色框：正常人体姿态。
- `fall score`：当前帧姿态几何得分。
- `temporal`：短时序平滑后的最终告警状态。

## Day1 交付物

1. `requirements.txt`：训练环境依赖。
2. `configs/fall_pose.yaml`：数据集配置模板。
3. `scripts/train_pose_baseline.py`：训练入口。
4. `scripts/infer_video.py`：视频推理 Demo。
5. `fall_detection/`：关键点跌倒判别核心代码。
6. `scripts/validate_dataset.py`：协助组员丙校验数据结构。
7. `scripts/scaffold_dataset.py`：创建数据集目录骨架。
