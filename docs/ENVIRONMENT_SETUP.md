# 部署环境搭建说明

## 1. 通用 ONNX 基准环境

推荐在 Windows 或 Ubuntu 上使用 Python 3.10/3.11：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-benchmark.txt
python tools/check_environment.py
```

该环境用于：

- ONNX合法性检查；
- 参数量和权重大小统计；
- CPU/CUDA ONNX Runtime基准测速；
- 生成统一JSON报告。

它不能替代真实NPU测速。

## 2. 当前初赛采用的通用NPU路线

团队目前没有实体开发板，也没有确定具体芯片型号，因此当前阶段不强制安装
RKNN、CANN、ATC或其他厂商SDK。

当前需要完成：

1. 导出固定输入Shape的ONNX；
2. 真实统计模型参数量和FP32权重；
3. 检查ONNX算子和动态Shape；
4. 测量PC端ONNX Runtime参考速度；
5. 估算FP32、FP16和INT8权重存储；
6. 设计INT8量化和端侧后处理方案；
7. 将NPU延迟和内存标记为设计目标，不写成实测值。

如果后续获得实体设备，再进入对应厂商的真实转换与测速阶段。

## 3. 可选的RKNN路线

RKNN环境必须根据目标芯片选择，例如RK3566、RK3568、RK3588等。

建议结构：

```text
Ubuntu x86_64 转换机：
  RKNN-Toolkit2 → ONNX 转 RKNN、量化和模拟器分析

目标开发板：
  RKNN-Toolkit-Lite2 / Runtime → 真实推理、延迟和内存测试
```

第1天需要团队确认并记录：

- 具体芯片型号；
- 开发板系统版本；
- RKNN SDK版本；
- Python版本；
- NPU驱动及Runtime版本；
- 是否能取得厂商提供的匹配wheel和示例工程。

不要在未确认芯片和SDK版本时随意安装网络上的RKNN wheel。

## 4. 可选的海思路线

海思部署环境同样依赖具体芯片和配套SDK。通常需要在厂商规定的Linux环境中
使用CANN/ATC或芯片专用转换工具，将ONNX转换为设备模型，再在目标板卡测量。

第1天需要确认：

- 海思具体芯片型号；
- 对应SDK/CANN版本；
- ATC是否可执行；
- 支持的ONNX opset及算子列表；
- 设备侧运行工具和性能分析工具；
- 板卡是否具备可访问权限。

## 5. 模型接口约定

组员甲首次导出的ONNX模型建议满足：

- 单输入；
- NCHW；
- batch固定为1；
- 输入尺寸固定为`1×3×384×384`；
- 输入类型为FP32或UINT8；
- 尽量使用标准Conv、ReLU/SiLU、Resize、Concat等算子；
- 避免动态Shape、动态控制流及自定义算子；
- 同时提供类别名称、归一化参数、颜色通道顺序和后处理说明。

## 6. 报告口径

至少分开记录：

1. 模型纯推理时间；
2. 图像预处理时间；
3. NMS/后处理时间；
4. 时序判别时间；
5. 完整端到端时间；
6. 主机进程内存；
7. NPU真实内存。

不能把ONNX Runtime的主机RSS写成NPU内存。

在没有实体设备时，报告应明确使用：

```text
参数量：ONNX实测
FP32权重：ONNX实测
PC推理时间：参考实测
INT8权重：依据参数量计算
NPU端到端延迟：设计目标，未上板验证
NPU运行内存：设计目标，未上板验证
```
