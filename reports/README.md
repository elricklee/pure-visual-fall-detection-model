# 报告目录

本目录保留当前候选模型的可复现证据。历史过程材料统一放入 `archive/`，不再作为当前结论入口。

## 目录结构

- `model/`：FP32/INT8模型审计、CPU参考测速、量化记录、输出一致性与NPU可行性。
- `evaluation/`：姿态、视频、伪红外、FP32/INT8一致性和多人功能验证。
- `environment/`：当前可复现运行环境。
- `compliance/`：赛题硬指标输入与检查结果。
- `summaries/`：供团队阅读和提交材料引用的人工总结。
- `documents/`：当前综合 Word 报告。
- `archive/day2_video_eval/`：旧 Day2 视频评估结果。
- `archive/project_reports/`：甲侧历史报告和训练状态追溯。

## 当前核心模型证据

- `model/fp32_model_audit.json`
- `model/fp32_cpu_benchmark_reference.json`
- `model/fp32_npu_feasibility.json`
- `model/int8_quantization.json`
- `model/int8_model_audit.json`
- `model/int8_cpu_benchmark.json`
- `model/int8_output_comparison.json`
- `model/int8_npu_feasibility.json`

## 当前评估证据

- `evaluation/fp32_pose_eval_visible.json`
- `evaluation/fp32_pose_eval_pseudo_ir.json`
- `evaluation/int8_pose_eval_visible.json`
- `evaluation/int8_pose_eval_pseudo_ir.json`
- `evaluation/fp32_video_eval_visible.json`
- `evaluation/fp32_video_eval_pseudo_ir.json`
- `evaluation/int8_video_eval_visible.json`
- `evaluation/int8_video_eval_pseudo_ir.json`
- `evaluation/int8_fp32_video_parity_visible.json`
- `evaluation/int8_fp32_video_parity_pseudo_ir.json`
- `evaluation/pseudo_ir_labeled_video_eval.json`
- `evaluation/multi_person_pipeline_eval.json`
- `evaluation/multi_person_functional_validation.json`

## 交付口径

- `cpu_benchmark` 仅代表当前电脑CPU参考速度，不代表NPU实测。
- `pseudo_ir` 是灰度复制的伪红外流程验证，不代表真实红外传感器数据。
- `parity` 表示两个模型版本的一致性；没有真值时不能解释为准确率。
- 多人合成视频只用于功能链路验证，不能作为真实多人场景准确率结论。
- 端侧推理不超过100ms、NPU运行内存不超过20MB仍是设计目标，待具体板卡实测。
- `archive/` 内文件只用于追溯版本演进，不优先引用。
