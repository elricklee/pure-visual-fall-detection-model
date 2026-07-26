# 7.26新权重FP32刷新与复测结果

执行日期：2026-07-26

## 本次范围

本次只执行：

1. 将`artifacts/pytorch/7.26_Train/best.pt`导出为版本化FP32 ONNX；
2. 验证PyTorch与ONNX输出等价性；
3. 重新执行模型审计；
4. 重新执行PC端ONNX Runtime参考测速；
5. 更新通用NPU部署可行性分析；
6. 使用同一环境复测旧ONNX，形成公平A/B对照。

未开始INT8量化及其他Day 3任务。

## 新版模型产物

```text
artifacts/onnx/yolov8n_pose_384_fp32_7_26.onnx
configs/model_yolov8n_pose_384_7_26.yaml
```

## ONNX审计

| 项目 | 新版结果 | 赛题要求 | 状态 |
|---|---:|---:|---|
| ONNX初始化元素 | 3.3142M | ≤20M | 通过 |
| ONNX初始化权重 | 13.2568MB | ≤80MB | 通过 |
| ONNX文件大小 | 13.3421MB | — | 记录 |
| 节点数 | 266 | — | 记录 |
| 输入 | `1×3×384×384` | 固定输入 | 通过 |
| 输出 | `1×56×3024` | — | 通过 |
| 非常见高风险算子 | 0 | — | 通过 |

ONNX初始化元素包含导出器固化的非训练常量，因此可能略高于PyTorch训练参数量，但不影响当前硬性指标结论。

## 等价性验证

| 项目 | 结果 |
|---|---:|
| 最大绝对误差 | 0.0006713867 |
| 平均绝对误差 | 0.0000082378 |
| 样例检测人数 | 1 |

结论：新版FP32 ONNX能够正确加载、推理，并与PyTorch输出保持一致。

## 同环境PC参考测速

环境：

- Windows 11；
- ONNX Runtime 1.28.0；
- CPUExecutionProvider；
- 输入`1×3×384×384`；
- 预热20次，正式运行100次；
- 新旧模型使用同一输入和同一环境。

| 指标 | 旧ONNX | 7.26新ONNX | 变化 |
|---|---:|---:|---:|
| 平均延迟 | 12.8848ms | 15.1697ms | +17.73% |
| P95延迟 | 14.9944ms | 20.7942ms | +38.68% |
| 平均FPS | 77.611 | 65.921 | -15.06% |
| 进程峰值RSS | 102.3345MB | 103.6370MB | +1.27% |
| ONNX文件大小 | 13.3123MB | 13.3421MB | +0.22% |

新版权重在本次同环境CPU测试中比旧版稍慢。不能仅凭此前269ms的旧报告进行比较，因为此前使用ONNX Runtime 1.27且系统抖动明显。

PC结果只用于版本回归，不代表真实NPU性能，也不包含视频解码、预处理、NMS、关键点解码、时序判断和报警输出。

## NPU可行性

- FP16权重估算：6.6284MB；
- INT8权重估算：3.3142MB；
- 固定Batch 1和固定384输入；
- 没有动态Shape；
- 没有自定义算子；
- NMS和时序逻辑位于模型外；
- 具体厂商兼容性仍需转换日志验证。

## 当前结论

**新版FP32 ONNX刷新成功，模型规模继续满足赛题要求，结构仍具备通用NPU部署基础。**

但在同环境CPU参考测试中，新版比旧版慢17.73%。在将新版设为最终基线前，应结合新版训练精度、红外表现和后续INT8结果综合判断，不能只根据“权重更新”直接替换旧版。

## 证据文件

- `reports/day3_fp32_model_audit.json`
- `reports/day3_fp32_benchmark.json`
- `reports/day3_fp32_old_baseline_benchmark_same_env.json`
- `reports/day3_fp32_npu_feasibility.json`
- `reports/day3_fp32_export_log.txt`
- `samples/day3_fp32_expected.jpg`
- `samples/day3_fp32_validation.json`
