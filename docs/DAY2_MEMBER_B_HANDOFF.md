# 组员乙 Day 2 完成交接单

## 核心任务

- [x] 接收并核验甲的第一版PyTorch权重；
- [x] 获得固定`1×3×384×384`测试ONNX；
- [x] 完成ONNX合法性与输入输出审计；
- [x] 完成参数量和FP32权重实测；
- [x] 完成ONNX算子组成与动态Shape检查；
- [x] 完成PC端ONNX Runtime参考测速；
- [x] 完成主机进程RSS记录；
- [x] 完成FP16、INT8权重规模分析；
- [x] 完成通用NPU部署可行性分析；
- [x] 完成赛题硬性指标检查；
- [x] 向甲输出性能短板和轻量化建议。

## 交叉辅助任务

- [x] 根据性能结果提出384固定输入、INT8校准和模型外后处理建议；
- [x] 给出多人时序状态隔离建议；
- [x] 完成PPT端侧部署和落地实用性板块文案；
- [x] 明确PC参考值、设计目标和真实NPU实测的区别。

## 本日主要结果

| 项目 | 结果 |
|---|---:|
| 参数量 | 3.29M |
| FP32初始化权重 | 13.1599MB |
| FP16权重估算 | 6.5799MB |
| INT8权重估算 | 3.29MB |
| ONNX输入 | `1×3×384×384` |
| ONNX输出 | `1×56×3024` |
| PC平均推理 | 269.1716ms |
| PC P95推理 | 589.4118ms |
| PC进程峰值RSS | 102.3304MB |
| 合规状态 | `design_ready_unverified` |

PC数据来自`CPUExecutionProvider`，不代表NPU性能。

## 产出文件

```text
artifacts/onnx/yolov8n_pose_384_fp32.onnx
artifacts/pytorch/yolov8n-pose.pt
configs/model_yolov8n_pose_384.yaml
reports/model_audit.json
reports/benchmark.json
reports/npu_feasibility.json
reports/compliance.json
reports/compliance_report.json
reports/member_b_day2_performance_report.md
reports/member_b_day2_feedback_to_a.md
docs/PPT_DAY2_EDGE_DEPLOYMENT.md
samples/day2_input.jpg
samples/day2_expected.jpg
samples/day2_expected.json
```

## 当前未验证项

- 真实NPU端到端延迟；
- 真实NPU运行内存；
- 具体海思或瑞芯微SDK转换兼容性；
- 独立测试集上的最终准确率；
- 充分的红外、微光和遮挡泛化能力。

以上项目不属于无实体板卡条件下的Day 2完成阻塞项，但必须在文档中保持“未验证”状态。

## 进入Day 3的建议

乙下一步应配合甲：

1. 准备INT8量化校准集；
2. 验证量化前后关键点和视频级效果；
3. 检查红外/灰度预处理的一致性；
4. 保持固定Shape和标准算子；
5. 为Day 4端到端耗时拆分与正式性能报告预留接口。

## 验收结论

**乙 Day 2 核心任务和交叉辅助任务已完成。**

当前模型规模达标、ONNX可运行、结构具备通用NPU部署基础。端侧100ms和NPU运行内存20MB仍为设计目标，不能写成真实实测结果。
