# 多人 track_id 状态隔离实现方案

## Context

当前跌倒检测系统只支持单人模式：每帧从所有检测结果中选一个"主人体"（`select_primary_pose_index`），用一个全局的 `FallDetector` + `TemporalFallStateMachine` 处理。多人场景下状态会串扰——A 跌倒了 B 站着，状态机被两人轮流喂数据，结果不可靠。

本次改动引入 `TrackerManager`，为每个 track_id 维护独立的 detector + state_machine，实现多人状态隔离。

## 核心设计原则

1. **不修改 `FallDetector` 和 `TemporalFallStateMachine`** — 它们本身就是有状态的，只需为每个 track_id 各建一个实例
2. **不破坏单人模式** — `--multi-person` 是新 flag，不传就走原有路径，所有现有测试不受影响
3. **复用 Ultralytics 内置 ByteTrack** — `model.track()` 自带跟踪，`result.boxes.id` 直接返回 track_id

## 实施步骤

### Step 1: 新增 `fall_detection/tracker_manager.py`

核心类 `TrackerManager`，管理 `track_id → (detector, state_machine)` 映射：

```python
@dataclass(frozen=True)
class PersonResult:
    track_id: int
    decision: FallDecision
    state_decision: TemporalStateDecision | None
    box_xyxy: Iterable[float]
    keypoints: Iterable[Iterable[float]]
    det_conf: float

class TrackerManager:
    def __init__(self, fall_config, state_config, stale_threshold=30, use_state_machine=True)
    def update(self, track_id, keypoints, box_xyxy, frame_height, det_conf) -> PersonResult
    def cleanup_stale(self) -> list[int]  # 清理离场人员
    def reset_person(self, track_id) -> None
    def reset_all(self) -> None
    @property active_ids -> list[int]
```

关键逻辑：
- `_get_or_create(track_id)` — 首次出现自动创建 detector + state_machine
- `_last_seen` 字典记录每个 ID 最后出现帧号，超 `stale_threshold` 帧未出现则清理
- `_frame_count` 由 `update()` 调用驱动

### Step 2: 更新 `fall_detection/__init__.py`

新增导出 `PersonResult`、`TrackerManager`。

### Step 3: 扩展 `fall_detection/visualization.py`

纯增量添加，原有函数不动：

- `PERSON_COLORS` — 8 色调色板，按 track_id 取模分配
- `_person_color(track_id)` — 取颜色
- `draw_multi_decision(image, box, decision, track_id, det_conf)` — 带人 ID 的 bbox 标签
- `draw_multi_status_banner(image, person_results)` — 多行状态横幅，每人一行

### Step 4: 改造 `scripts/infer_video.py`

**CLI 新增参数：**
- `--multi-person` — 启用多人追踪模式
- `--tracker` — 追踪器配置，默认 `bytetrack.yaml`
- `--stale-threshold` — 离场清理帧数阈值，默认 30

**新增 `process_frame_multi()`：**
- 使用 `model.track()` 替代 `model.predict()`
- 从 `result.boxes.id` 获取 track_id
- 对每个检测到的人调用 `tracker_mgr.update()`
- 调用 `draw_multi_decision()` + `draw_multi_status_banner()`
- 每帧调用 `cleanup_stale()`

**`main()` 分支：**
- `--multi-person` 未指定 → 原有单人路径完全不变
- `--multi-person` 指定 → 创建 `TrackerManager`，帧循环用 `model.track(persist=True)`

### Step 5: 改造 `scripts/evaluate_videos.py`

与 infer_video.py 同理：
- 新增 `--multi-person`、`--tracker`、`--stale-threshold` 参数
- 多人模式下用 `TrackerManager`，"任何一个人确认跌倒"即视为该视频预测为 fall

### Step 6: 新增 `tests/test_tracker_manager.py`

测试用例：
- 首次出现的 track_id 自动创建追踪器
- 同一 track_id 连续帧复用同一 detector/state_machine
- 不同 track_id 状态完全隔离
- cleanup_stale 清理过期 ID
- reset_person / reset_all 正确重置
- vote 模式下 state_decision 为 None
- active_ids 返回所有活跃 ID

### Step 7: 扩展 `tests/test_visualization.py`

新增多人可视化测试：`draw_multi_decision` 标签含 track_id、跌倒用红色、`draw_multi_status_banner` 空列表和多人都正确绘制。

## 文件清单

| 文件 | 动作 | 改动概要 |
|------|------|----------|
| `fall_detection/tracker_manager.py` | 新增 | PersonResult + TrackerManager，约 100 行 |
| `fall_detection/__init__.py` | 修改 | 新增 2 个导出 |
| `fall_detection/visualization.py` | 修改 | 新增 3 函数 + 颜色常量，原函数不动 |
| `scripts/infer_video.py` | 修改 | process_frame_multi + CLI 参数 + main 分支 |
| `scripts/evaluate_videos.py` | 修改 | 多人评估路径 + CLI 参数 |
| `tests/test_tracker_manager.py` | 新增 | 7+ 测试用例 |
| `tests/test_visualization.py` | 修改 | 新增多人可视化测试 |

**不改动的文件：** `detection_selection.py`、`state_machine.py`、`fall_logic.py`、`pose_features.py`、`image_utils.py`、`test_detection_selection.py`、`test_state_machine.py`、`test_fall_logic.py`

## 验证方式

1. `pytest tests/` — 全部测试通过（含原有单人测试 + 新增多人测试）
2. 单人模式回归：`python scripts/infer_video.py --source <video> --model <model>` 行为与改动前一致
3. 多人模式端到端：`python scripts/infer_video.py --source <multi_person_video> --model <model> --multi-person --roi full` 输出视频中不同人有不同颜色标签和独立状态
4. 多人评估：`python scripts/evaluate_videos.py --source <dir> --model <model> --multi-person` 正常产出评估报告
