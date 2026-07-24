# 组员乙 Day 1：端侧部署与评测工具包

本目录覆盖分工方案中组员乙第 1 天的核心任务：

- 部署环境自检；
- ONNX Runtime 基准测速；
- 模型参数量、权重大小和算子审计；
- 推理耗时与进程内存统计；
- 无实体板卡条件下的通用NPU部署可行性分析；
- 赛题硬性指标自动校验；
- 给组员甲、丙的接口和交接清单。

## 目录

```text
member_b_day1_toolkit/
├─ configs/
│  └─ compliance.example.json
├─ docs/
│  ├─ DAY1_HANDOFF.md
│  └─ ENVIRONMENT_SETUP.md
├─ tools/
│  ├─ benchmark_onnx.py
│  ├─ analyze_npu_feasibility.py
│  ├─ check_compliance.py
│  ├─ check_environment.py
│  └─ inspect_onnx.py
├─ tests/
│  └─ test_tools.py
└─ requirements-benchmark.txt
```

## 快速开始

建议使用 Python 3.10 或 3.11 创建独立环境。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-benchmark.txt
```

先检查环境：

```powershell
python tools/check_environment.py --output reports/environment.json
```

拿到组员甲导出的 ONNX 模型后：

```powershell
python tools/inspect_onnx.py models/fall_detector.onnx `
  --output reports/model_audit.json

python tools/benchmark_onnx.py models/fall_detector.onnx `
  --warmup 20 `
  --runs 100 `
  --output reports/benchmark.json

python tools/analyze_npu_feasibility.py reports/model_audit.json `
  --benchmark reports/benchmark.json `
  --output reports/npu_feasibility.json
```

把模型审计结果、测速结果及实测 NPU 数据填入一份指标文件后执行：

```powershell
Copy-Item configs/compliance.example.json reports/compliance.json
python tools/check_compliance.py reports/compliance.json `
  --output reports/compliance_report.json
```

返回码说明：

- `0`：全部硬性指标均有实测证据并通过；
- `2`：存在超标项或缺失项；
- `3`：方案目标满足要求，但NPU延迟或内存尚未上板验证。

## 当前边界

ONNX Runtime测得的是当前计算机上的模型推理耗时和进程RSS，不能替代真实
NPU的耗时及NPU内存数据。当前初赛路线不要求团队拥有实体板卡，因此工具会将
相关指标标记为“设计目标、未验证”，不会伪装成实测通过。
