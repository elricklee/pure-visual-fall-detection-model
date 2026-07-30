# 修改日志

> **日志格式**：时间 | 修改目的（原因） | 改动文件 | 具体改动（删除行 / 新增行）

---

## 2026-07-25

**目的**：修复 `infer_video.py` 非红外模式下双重 YOLO 推理导致的进程卡死问题（`model.predict(stream=True)` 内部推理一遍 + `process_frame()` 再推理一遍，造成阻塞，res.mp4 已生成但无法完成写入无法打开）。

**文件**：[scripts/infer_video.py](scripts/infer_video.py)

**具体改动**：

- **删除行 149-190（原红外/非红外两分支）**：
  - 删红外分支（原 149-169）：`if args.infrared: cap = cv2.VideoCapture(...)` 整段
  - 删非红外分支（原 171-190）：`else: for result in model.predict(source=..., stream=True, ...)` 整段（含双重推理逻辑）

- **新增行 149-179（统一 cv2 读帧路径）**：
  - 行 149-160：`cv2.VideoCapture` 统一读取视频，获取 fps/宽/高，创建 `VideoWriter`
  - 行 162-174：单层 `while True` 循环用 `cap.read()` 逐帧读取，按 `args.infrared` 开关决定 `display_frame` 来源（红外转换或原帧 copy），每帧仅调用一次 `process_frame()`
  - 行 176：`cap.release()` 释放视频源

---

**目的**：修复 `infer_video.py` 中 `select_primary_pose_index()` 返回值类型错误（函数返回 `tuple[int, ndarray, ndarray]`，原代码直接当 `int` 用，导致 `IndexError: arrays used as indices must be of integer (or boolean) type`）。

**文件**：[scripts/infer_video.py](scripts/infer_video.py)

**具体改动**：

- **修改行 82-87**：
  - 删：`selected_idx = select_primary_pose_index(...)` → `if selected_idx is not None: box = boxes[selected_idx]`
  - 改为：`sel_result = select_primary_pose_index(...)` → `if sel_result is not None: selected_idx = sel_result[0]; box = sel_result[1][selected_idx]; kpts = sel_result[2][selected_idx]`（正确解包返回元组）
