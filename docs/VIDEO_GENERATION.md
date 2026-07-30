# 视频 Demo 生成命令

本文档用于本地生成跌倒检测可视化视频。后续模型或算法更新后，由成员在本机
执行命令，不将生成的视频提交到 Git。

## 1. 进入项目并激活环境

```powershell
cd path\to\pure-visual-fall-detection-model
.\.venv\Scripts\Activate.ps1
```

可选：将 Ultralytics 和 Matplotlib 配置目录放在项目内。

```powershell
$env:YOLO_CONFIG_DIR="$PWD\Ultralytics"
$env:MPLCONFIGDIR="$PWD\.matplotlib"
```

## 2. 生成单个视频

```powershell
python scripts\infer_video.py `
  --source datasets\fall_pose\videos\raw\urfall_fall_02_cam0.mp4 `
  --model runs\pose\runs\train\fall_pose_day2_v1\weights\best.pt `
  --output runs\demo\manual\urfall_fall_02_demo.mp4 `
  --device 0 `
  --decision-mode state_machine `
  --roi auto
```

参数说明：

- `--source`：输入视频路径
- `--model`：模型权重路径
- `--output`：输出视频路径
- `--device 0`：使用第 1 张 NVIDIA GPU
- `--decision-mode state_machine`：使用姿态时序状态机
- `--roi auto`：宽屏双画面自动只选右侧 RGB，普通视频使用完整画面

## 3. UR Fall 左右双画面

UR Fall 视频左侧是深度图、右侧是 RGB 图。可以显式指定只处理右侧：

```powershell
python scripts\infer_video.py `
  --source datasets\fall_pose\videos\raw\urfall_fall_01_cam0.mp4 `
  --model runs\pose\runs\train\fall_pose_day2_v1\weights\best.pt `
  --output runs\demo\manual\urfall_fall_01_demo.mp4 `
  --device 0 `
  --decision-mode state_machine `
  --roi right
```

普通手机视频建议使用：

```text
--roi full
```

## 4. 批量生成目录下所有视频

```powershell
$model = "runs\pose\runs\train\fall_pose_day2_v1\weights\best.pt"
$sourceDir = "datasets\fall_pose\videos\raw"
$outputDir = "runs\demo\manual_batch"

New-Item -ItemType Directory -Force -Path $outputDir

$videos = Get-ChildItem $sourceDir -Filter *.mp4
foreach ($video in $videos) {
    $output = Join-Path $outputDir ($video.BaseName + "_demo.mp4")

    python scripts\infer_video.py `
      --source $video.FullName `
      --model $model `
      --output $output `
      --device 0 `
      --decision-mode state_machine `
      --roi auto
}
```

## 5. 调整状态机和跌倒阈值

默认参数：

```text
fall_score_threshold = 0.70
temporal_window = 5
temporal_votes = 3
```

画面含义：

- `fall_score`：由框宽高比、躯干角度和肩髋间距计算的连续风险分数
- `FALL_POSE`：当前单帧姿态超过阈值，但尚未完成时序确认
- `FALL`：状态机已确认连续跌倒
- `STATE`：完整事件状态，包含正常、疑似、确认和恢复

手动调整示例：

```powershell
python scripts\infer_video.py `
  --source path\to\input.mp4 `
  --model runs\pose\runs\train\fall_pose_day2_v1\weights\best.pt `
  --output runs\demo\manual\output.mp4 `
  --device 0 `
  --decision-mode state_machine `
  --fall-score-threshold 0.70 `
  --temporal-window 5 `
  --temporal-votes 3 `
  --roi full
```

调参建议：

- 误报较多：提高 `--fall-score-threshold`，例如 `0.75`
- 漏报较多：降低到 `0.65`
- 短暂误报：增加 `--temporal-votes`
- 报警太慢：减少 `--temporal-votes`

## 6. 回退旧的 5 帧投票逻辑

不使用状态机时：

```text
--decision-mode vote
```

完整示例：

```powershell
python scripts\infer_video.py `
  --source path\to\input.mp4 `
  --model runs\pose\runs\train\fall_pose_day2_v1\weights\best.pt `
  --output runs\demo\manual\vote_output.mp4 `
  --device 0 `
  --decision-mode vote `
  --roi full
```

## 7. 画面含义

- `STATE: NORMAL`：正常状态
- `STATE: SUSPECT_FALL`：出现下落或疑似跌倒姿态
- `STATE: FALL_CONFIRMED`：连续多帧确认跌倒
- `STATE: RECOVERY`：从跌倒状态恢复
- `score`：姿态几何连续评分，不是 YOLO 检测置信度

## 8. 输出管理

视频和模型权重已被 `.gitignore` 忽略：

```text
runs/
*.mp4
*.pt
```

生成视频只保存在本地，不执行 `git add -f` 强制提交。
