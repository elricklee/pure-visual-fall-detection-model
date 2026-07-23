# Day1 数据准备说明

目标是把原始视频整理成 `YOLOv8 pose` 可训练的数据集。

## 最终目录

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
  videos/
    raw/
    demo/
```

## 先做什么

### 1. 建目录骨架

```powershell
python scripts\scaffold_dataset.py --root datasets\fall_pose
```

### 2. 把原始视频放进去

把你们收集到的可见光视频、跌倒视频、正常行走视频，先放到：

```text
datasets/fall_pose/videos/raw/
```

### 2.1 拍什么视频

先拍真实场景里能覆盖关键姿态的视频，不要只拍“摔倒瞬间”。

建议拍这些场景：

1. 正常站立
2. 走路
3. 坐下
4. 弯腰
5. 蹲下
6. 慢慢躺下
7. 真实跌倒动作
8. 跌倒后躺地

这样后面模型更容易区分“正常弯腰/坐下”和“真正跌倒”。

### 2.2 上哪里找视频

优先顺序：

1. 自己手机拍
2. 组员互拍
3. 公开视频素材

如果是公开视频，只选清晰、无遮挡、人物完整、角度稳定的视频。不要拿电影片段、监控压缩很重的素材，后面标注会很痛苦。

### 2.3 每个类型拍几个视频

先做 Day1 MVP，建议最小集：

- 正常动作：4 到 6 个视频
- 跌倒动作：4 到 6 个视频
- 总时长：每个视频 10 到 20 秒

如果时间紧，至少保证：

- 正常 3 个
- 跌倒 3 个

### 3. 抽帧

把视频按固定间隔抽帧，优先保留动作变化明显的片段。

建议：

- 普通视频：每 5 到 10 帧取 1 帧
- 动作剧烈视频：每 2 到 5 帧取 1 帧

推荐先用：

```powershell
python scripts\extract_frames.py --source datasets\fall_pose\videos\raw --output datasets\fall_pose\images\train --every-n 5
```

如果某段动作特别密集，可以改成 `--every-n 3`。

抽出来的图片先放到：

```text
datasets/fall_pose/images/train/
datasets/fall_pose/images/val/
datasets/fall_pose/images/test/
```

### 4. 标注

每张图都要有同名 `.txt` 标签文件，放在对应 `labels` 目录下。

YOLO pose 单行格式：

```text
class x_center y_center width height kpt_x kpt_y visible ...
```

COCO 17 关键点顺序不变，坐标都用 0 到 1 归一化值。

### 5. 先做小样本集

Day1 不要一口气做大集，先做一个小而干净的版本：

- 6 到 12 个视频
- 300 到 800 张图
- 标注全部正确

先跑通训练和推理，再扩充规模。

### 6. 校验

```powershell
python scripts\validate_dataset.py --data configs\fall_pose.yaml
```

重点看：

- 图片和标签是否一一对应
- 关键点数量是否是 17
- train / val 是否都有样本

## 组员甲今天该配合什么

1. 帮组员丙确认抽帧后图片格式统一。
2. 检查标签是否符合 pose 格式。
3. 用 `validate_dataset.py` 先把结构问题排掉。
