# 组员乙：INT8量化与统一可见光/红外流程验证报告

验证日期：2026-07-26；文件整理与复测日期：2026-07-28

## 完成结论

乙第三天计划内的技术任务已完成：

1. 以甲的 `7.26_Train/best.pt` 对应 FP32 ONNX 为源模型；
2. 建立可复现的静态 INT8 QDQ 量化脚本；
3. 使用 64 个校准输入完成逐通道量化，其中可见光 24 个、伪红外 40 个；
4. 完成 FP32/INT8 的模型体积、PC 参考速度、输出误差和姿态指标对比；
5. 完成可见光/伪红外统一视频流程验证；
6. 完成 ONNX 算子审计和通用 NPU 转换风险说明。

正式候选模型：

```text
artifacts/onnx/fall_pose_384_int8_qdq.onnx
SHA256 31636F87F7654E172EE9BC4A72B5F6B4CC3875D9285D439BCE95042146B8EE1C
```

## 量化方案

| 项目 | 配置 |
|---|---|
| 量化方法 | 静态量化 |
| 格式 | QDQ |
| 校准方法 | MinMax |
| 激活 | QUInt8 |
| 权重 | QInt8，逐通道 |
| 量化算子 | Conv |
| 输入 | `1×3×384×384` FP32 |
| 输出 | `1×56×3024` FP32 |
| ONNX opset | 12 → 13 |
| 校准样本 | 24 可见光＋40 伪红外 |

逐通道 QDQ 的 `axis` 属性需要 opset 13，因此量化模型升级到 opset 13。模型不包含自定义算子，NMS和跌倒时序判别仍放在模型外。

伪红外数据由可见光图像灰度化后复制为三通道，只能证明输入和处理链路兼容，不能替代真实红外传感器数据。

## 模型体积与 PC 参考速度

环境：Windows 11、ONNX Runtime 1.28.0、CPUExecutionProvider、固定输入，预热 20 次、正式运行 100 次。

| 指标 | FP32 | INT8 | 变化 |
|---|---:|---:|---:|
| ONNX文件 | 13.3421MB | 3.6931MB | 缩小72.32% |
| 平均延迟 | 24.3201ms | 19.0201ms | 降低21.79% |
| P95延迟 | 44.7266ms | 28.7053ms | 降低35.82% |
| FPS | 41.118 | 52.576 | 提升27.87% |
| 进程峰值RSS | 103.1578MB | 100.1554MB | 降低2.91% |

以上是 PC CPU 参考值，不是 NPU 实测；也不包含视频解码、预处理、NMS、时序判断和报警输出。

## 精度回归

本地验证集只有 11 张图，以下数据只用于版本回归，不代表最终比赛精度。

| 输入 | 指标 | FP32 | INT8 | 变化 |
|---|---|---:|---:|---:|
| 可见光 | Pose mAP50 | 0.9950 | 0.9950 | 0 |
| 可见光 | Pose mAP50-95 | 0.53376 | 0.54595 | +1.22个百分点 |
| 伪红外 | Pose mAP50 | 0.9950 | 0.9950 | 0 |
| 伪红外 | Pose mAP50-95 | 0.39625 | 0.39577 | -0.05个百分点 |

22 个可见光/伪红外输入的 FP32 与 INT8 原始输出平均余弦相似度为 0.999707。

## 视频级一致性

当前 10 段视频没有可解析的跌倒/正常真值，因此只能统计版本一致性，不能计算准确率。

| 模式 | 告警结论一致率 | 首次告警平均帧差 | 说明 |
|---|---:|---:|---|
| 可见光 | 10/10（100%） | 0.3帧 | FP32与INT8一致 |
| 伪红外 | 9/10（90%） | 1.11帧 | 1段边界样本不同 |

伪红外差异发生在 `20240912_102330.mp4`。FP32只产生1个时序告警帧，INT8未触发；由于没有真值，不能判断这是漏报还是消除了边界误报，应在甲/丙补充视频标签后复核。

## NPU部署判断

- 模型文件3.6931MB，参数规模和权重存储目标有充足余量；
- 固定Batch 1、固定384输入，无动态Shape；
- 主体仍是Conv、Resize、Concat、Sigmoid等常见算子；
- 新增QDQ的`QuantizeLinear/DequantizeLinear`，需由具体芯片SDK确认能否融合为INT8 Conv；
- 当前没有实体板卡，NPU端到端延迟和运行内存仍是未验证项；
- 不得把19.0201ms的PC CPU结果写成NPU实测。

## 后续联调建议

1. 由甲、丙给10段现有视频补充跌倒/正常真值；
2. 对伪红外边界视频复核误报/漏报属性；
3. 获得真实红外样本后替换伪红外校准数据；
4. 确定目标板卡后，用厂商SDK验证QDQ融合、转换日志、端到端延迟和NPU运行内存；
5. 若真实红外精度下降超过2个百分点，再进行QAT或分层混合精度。

## 证据文件

- `reports/model/int8_quantization.json`
- `reports/model/int8_model_audit.json`
- `reports/model/int8_cpu_benchmark.json`
- `reports/model/fp32_cpu_benchmark_reference.json`
- `reports/evaluation/int8_pose_eval_visible.json`
- `reports/evaluation/int8_pose_eval_pseudo_ir.json`
- `reports/model/int8_output_comparison.json`
- `reports/evaluation/int8_fp32_video_parity_visible.json`
- `reports/evaluation/int8_fp32_video_parity_pseudo_ir.json`
- `reports/model/int8_npu_feasibility.json`

结论：**乙的INT8量化与验证任务已完成，INT8候选可进入端到端联调；真实红外与真实NPU指标仍需后续验证。**
