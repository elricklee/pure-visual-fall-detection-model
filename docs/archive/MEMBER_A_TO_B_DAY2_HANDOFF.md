# 甲 → 乙：Day 2 开工前模型交接资料

## 1. 交接目的

乙在 Day 2 的核心工作是接收甲的第一版模型，完成参数量、权重大小、ONNX 算子和 PC 参考性能评测，并向甲反馈轻量化问题。

因此，甲应在乙 Day 2 开工前提供一套“可运行、可审计”的基线模型资料。该模型不要求达到最终精度，也不等同于 Day 4 的正式优化模型。

## 2. 必交资料

### 2.1 第一版测试 ONNX

建议文件：

```text
artifacts/onnx/fall_pose_384_fp32.onnx
```

最低要求：

- 固定输入 Shape：`1×3×384×384`；
- Batch 固定为 1；
- 输入布局：NCHW；
- 输入精度：FP32；
- 关闭动态 Shape；
- 不把 NMS 和时序跌倒逻辑封装进模型；
- 优先使用标准 ONNX 算子；
- 能通过 ONNX 合法性检查；
- 能在 ONNX Runtime CPU 后端完成至少一次推理。

如果 384 输入暂时无法导出，甲应提前说明原因，并提供当前可运行的固定输入版本，不得只提供动态 Shape 模型。

### 2.2 对应的 PyTorch 权重

建议文件：

```text
artifacts/pytorch/yolov8n-pose.pt
```

可以提供以下任一种：

- 甲首轮训练得到的权重；
- 当前实际使用的微调权重；
- 尚未完成跌倒数据训练时使用的 COCO 预训练权重。

如果使用的是 COCO 预训练权重，必须明确标注：

> 当前模型负责人体框和 17 个关键点检测，`fall/normal` 由关键点几何规则和时序后处理产生，不是模型原生类别。

### 2.3 模型接口说明

建议文件：

```text
configs/fall_pose_384_fp32.yaml
```

至少填写：

```yaml
model_name: yolov8n-pose
model_stage: day2_baseline

input:
  name: 待甲填写
  shape: [1, 3, 384, 384]
  layout: NCHW
  dtype: float32
  color_order: RGB
  resize: letterbox
  scale: 0.00392156862745098
  mean: [0.0, 0.0, 0.0]
  std: [1.0, 1.0, 1.0]

output:
  names: [待甲填写]
  shapes: [待甲填写]
  model_class: person
  keypoint_count: 17
  keypoint_order: COCO

postprocess:
  confidence_threshold: 待甲填写
  nms_iou_threshold: 待甲填写
  nms_in_onnx: false
  coordinate_restore: 待甲填写
  fall_logic: fall_detection/
  temporal_window: 5
```

如果实际预处理方式与模板不同，应以代码真实行为为准修改，不得照抄错误参数。

### 2.4 ONNX 导出记录

建议文件：

```text
reports/model/fp32_model_audit.json
```

至少记录：

- 导出日期；
- Python 版本；
- PyTorch 版本；
- Ultralytics 版本；
- ONNX 版本；
- ONNX opset；
- 完整导出命令或脚本路径；
- 导出是否启用 simplify；
- 导出是否关闭 dynamic；
- ONNX Runtime 单次推理是否成功；
- 导出过程中出现的警告或已知问题。

### 2.5 测试输入与预期结果

建议文件：

```text
samples/reference_input.jpg
samples/reference_output_fp32.jpg
```

要求：

- 输入图片可由团队内部用于复现；
- 图片中至少包含一个完整可见人体；
- 预期结果应显示人体框和关键点；
- 如果同时展示跌倒判断，应注明该结果来自后处理逻辑；
- 如果不便提供结果图片，可以提供等价的 JSON 或 TXT 输出。

### 2.6 当前训练状态

建议文件：

```text
reports/archive/project_reports/training_status.md
```

至少说明：

- 当前使用预训练基线还是自建跌倒数据训练权重；
- 已完成的训练轮数；
- 训练输入尺寸；
- 使用的数据配置文件；
- 当前已有的验证指标；
- 尚未完成的训练工作；
- 已知误报、漏报和场景限制；
- 多人场景是否已经按人员 ID 隔离时序状态。

## 3. 推荐交付目录

```text
artifacts/
  pytorch/
    yolov8n-pose.pt
  onnx/
    fall_pose_384_fp32.onnx
configs/
  fall_pose_384_fp32.yaml
reports/
  model/fp32_model_audit.json
  training_status.md
samples/
  reference_input.jpg
  reference_output_fp32.jpg
```

## 4. 甲交付前自检

- [ ] ONNX 文件不是空文件；
- [ ] ONNX 输入 Shape 固定；
- [ ] ONNX Batch 为 1；
- [ ] ONNX 能在 CPUExecutionProvider 下完成一次推理；
- [ ] `.pt` 和 ONNX 来自同一模型版本；
- [ ] 接口说明与实际代码一致；
- [ ] 测试样例能够复现；
- [ ] 已明确模型类别是 `person`，跌倒状态来自后处理；
- [ ] 已标明当前训练状态和已知限制；
- [ ] 文件名、模型版本和导出记录能够相互对应。

## 5. 乙收到后的验收工作

乙收到交接包后负责：

1. 检查 ONNX 合法性、输入输出和算子组成；
2. 实测模型参数量和 FP32 权重大小；
3. 执行 ONNX Runtime CPU 参考测速；
4. 记录模型纯推理延迟和主机进程 RSS；
5. 分析 FP16、INT8 权重存储规模；
6. 检查动态 Shape、复杂算子及端侧转换风险；
7. 输出通用 NPU 部署可行性报告；
8. 向甲反馈 Day 2 需要优先优化的问题。

## 6. 当前阶段不要求提供

- 正式最终精度模型；
- Day 4 的完整端到端联调版本；
- 实体海思或瑞芯微开发板；
- RKNN、CANN/ATC 等厂商 SDK；
- 真实 NPU 延迟和运行内存；
- 同时适配多个 NPU 品牌。

## 7. 接收结论

满足以下条件即可允许乙开始 Day 2 工作：

- 测试 ONNX 可以加载和推理；
- `.pt` 权重与 ONNX 版本对应；
- 输入预处理和输出结构已经说明；
- 有一组可复现的输入与预期结果；
- 当前训练状态和模型限制已经如实记录。

仅提供 `.pt` 权重而没有测试 ONNX，不能满足乙现有 ONNX 审计和测速工具的输入要求，应视为交接资料不完整。
