# 无实体板卡的通用NPU部署可行性路线

## 路线结论

初赛阶段不指定海思或瑞芯微具体型号，不声称完成真实NPU部署。当前目标是用
可复现证据证明模型“足够轻、结构适合转换、具备进一步部署可行性”。

## 当前要真实完成的项目

| 项目 | 当前证据 |
|---|---|
| 模型参数量 | 从ONNX初始化张量真实统计 |
| FP32权重大小 | 从ONNX真实统计 |
| 输入输出Shape | 从ONNX真实读取 |
| 算子组成 | 从ONNX真实读取 |
| PC推理速度 | ONNX Runtime参考实测 |
| FP16/INT8权重体积 | 按参数位宽计算 |
| 可见光和红外 | 用测试数据验证 |

## 当前只能作为设计目标的项目

| 项目 | 目标 | 说明 |
|---|---:|---|
| NPU端到端延迟 | ≤100ms | 没有实体板卡，不能写成实测 |
| NPU运行内存 | ≤20MB | 参数存储不等于运行时总内存 |
| 厂商算子兼容 | 全部可转换 | 需确定芯片后用厂商SDK验证 |

## 模型设计约束

- 固定batch为1；
- 固定输入尺寸，优先`1×3×384×384`；
- 使用标准ONNX算子；
- 避免动态Shape和控制流；
- NMS、关键点解码和时序逻辑尽量放在NPU模型外；
- 参数量内部目标不超过10M；
- 预留INT8量化流程；
- 同时记录可见光和灰度/红外预处理。

## 推荐执行命令

```powershell
python tools/inspect_onnx.py artifacts/onnx/fall_pose_384_fp32.onnx `
  --output reports/model/fp32_model_audit.json

python tools/benchmark_onnx.py artifacts/onnx/fall_pose_384_fp32.onnx `
  --warmup 20 --runs 100 `
  --input-npy samples/reference_input_384.npy `
  --output reports/model/fp32_cpu_benchmark_reference.json

python tools/analyze_npu_feasibility.py reports/model/fp32_model_audit.json `
  --benchmark reports/model/fp32_cpu_benchmark_reference.json `
  --output reports/model/fp32_npu_feasibility.json

python tools/check_compliance.py reports/compliance/model_compliance_metrics.json `
  --output reports/compliance/model_compliance_report.json
```

## PPT建议表述

> 本方案面向海思、瑞芯微等常见端侧NPU平台设计，采用固定输入Shape、标准
> ONNX算子及INT8量化方案。当前已完成参数规模、模型存储、算子组成和PC参考
> 性能验证；端侧推理耗时≤100ms、NPU存储占用≤20MB作为设计目标，实际结果
> 需在确定目标板卡后进一步验证。

## 禁止使用的表述

- “海思NPU实测耗时XX毫秒”；
- “NPU内存实测XX MB”；
- “已经适配全部海思和瑞芯微芯片”；
- “PC端ONNX Runtime速度就是NPU速度”。

在没有实体设备和厂商转换日志时，上述说法都缺少证据。
