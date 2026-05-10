# ONNX与模型导出 - 模型压缩与部署


## 1. ONNX格式概述


ONNX（Open Neural Network Exchange）是一个开放的模型表示格式，旨在实现不同深度学习框架之间的模型互操作。ONNX定义了一套标准化的算子集和计算图表示，使模型可以在PyTorch训练后导出，在TensorRT、ONNX Runtime等推理引擎上运行。


### ONNX核心组成


- **计算图（Graph）：**
   由节点（Node）组成的有向无环图，描述计算流程
- **算子（Operator）：**
   标准算子集（如Conv、MatMul、Relu），覆盖主流神经网络操作
- **张量（Tensor）：**
   模型权重和中间计算结果的数据容器
- **元数据（Metadata）：**
   模型版本、生产者信息等


| 特性 | PyTorch (.pt/.pth) | ONNX (.onnx) |
| --- | --- | --- |
| 框架依赖 | 必须PyTorch | 框架无关，跨平台 |
| 推理引擎 | PyTorch推理 | ONNX Runtime / TensorRT / OpenVINO |
| 优化能力 | 有限 | 图优化、算子融合、常量折叠 |
| 部署端 | 服务器、移动端 | 服务器、移动端、嵌入式、浏览器 |
| 动态维度 | 原生支持 | 通过动态轴支持 |


## 2. PyTorch导出ONNX


### 基础导出


```
import torch
import torch.nn as nn
import torchvision.models as models

# 加载预训练模型
model = models.resnet18(pretrained=True)
model.eval()

# 创建示例输入
dummy_input = torch.randn(1, 3, 224, 224)

# 导出ONNX模型
torch.onnx.export(
    model,                       # 要导出的模型
    dummy_input,                 # 示例输入（用于追踪计算图）
    "resnet18.onnx",             # 输出文件路径
    export_params=True,          # 导出训练好的参数
    opset_version=14,            # ONNX算子集版本
    do_constant_folding=True,    # 常量折叠优化
    input_names=['input'],       # 输入节点名称
    output_names=['output'],     # 输出节点名称
    dynamic_axes={               # 动态轴配置
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
print("ONNX模型导出成功！")
```


### 动态轴 vs 静态轴


| 类型 | 描述 | 适用场景 |
| --- | --- | --- |
| 静态轴 | 所有维度固定，如 [1,3,224,224] | 固定输入尺寸的服务端部署 |
| 动态轴 | 指定维度可变，如 batch_size 动态 | 需处理不同batch size的场景 |
| 动态shape | 所有维度均可变 | 处理不同分辨率图像 |


> **Note:** 动态轴会略微影响推理性能，因为推理引擎无法针对固定尺寸做最优优化。生产环境中建议固定batch size和输入尺寸。


## 3. ONNX Runtime推理加速


```
import onnxruntime as ort
import numpy as np

# 创建推理会话
session = ort.InferenceSession(
    "resnet18.onnx",
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)

# 查看模型输入输出信息
for inp in session.get_inputs():
    print(f"输入: {inp.name}, 形状: {inp.shape}, 类型: {inp.type}")
for out in session.get_outputs():
    print(f"输出: {out.name}, 形状: {out.shape}, 类型: {out.type}")

# 准备输入数据
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# 运行推理
input_name = session.get_inputs()[0].name
output = session.run(None, {input_name: input_data})
predictions = output[0]
print(f"预测结果形状: {predictions.shape}")

# 性能基准测试
import time
num_runs = 100
start = time.time()
for _ in range(num_runs):
    session.run(None, {input_name: input_data})
elapsed = time.time() - start
print(f"平均推理时间: {elapsed/num_runs*1000:.2f} ms")
```


### ONNX Runtime后端选项


| 后端 | 硬件 | 适用场景 |
| --- | --- | --- |
| CUDAExecutionProvider | NVIDIA GPU | 高性能服务端推理 |
| CPUExecutionProvider | CPU | 通用部署、无GPU环境 |
| TensorrtExecutionProvider | NVIDIA GPU + TensorRT | 极致推理速度 |
| OpenVINOExecutionProvider | Intel CPU/GPU/VPU | Intel硬件优化 |
| CoreMLExecutionProvider | Apple设备 | iOS/macOS部署 |


## 4. ONNX模型优化


### 图优化级别


| 优化级别 | 包含优化 | 效果 |
| --- | --- | --- |
| Basic（基本） | 冗余节点消除、常量折叠 | 通用优化，无精度损失 |
| Extended（扩展） | 算子融合、数据布局优化 | 进一步加速 |
| Layout（布局） | NCHW→NHWC布局转换 | 针对GPU优化数据布局 |
| All（全部） | 所有优化的组合 | 最大加速效果 |


### 常见算子融合


- **Conv + BatchNorm + ReLU → FusedConvBNReLU：**
   将三个算子合并为一个，减少内存访问
- **MatMul + Add → Gemm：**
   通用矩阵乘法融合
- **连续Reshape消除：**
   移除冗余的形状变换操作


```
import onnx
from onnxruntime.transformers import optimizer

# 使用ONNX Runtime优化器
optimized_model = optimizer.optimize_model(
    "resnet18.onnx",
    model_type='bert',  # 或 'gpt2', 'vit' 等
    optimization_options=None,
    use_gpu=True
)
optimized_model.save_model_to_file("resnet18_optimized.onnx")
```


## 5. ONNX模型验证与调试


```
import onnx
import onnxruntime as ort
import numpy as np

# 1. 验证ONNX模型结构
model = onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
print("ONNX模型结构验证通过！")

# 2. 对比PyTorch和ONNX输出
import torch
import torchvision.models as models

torch_model = models.resnet18(pretrained=True).eval()
dummy_input = torch.randn(1, 3, 224, 224)

# PyTorch推理
with torch.no_grad():
    torch_output = torch_model(dummy_input).numpy()

# ONNX推理
session = ort.InferenceSession("resnet18.onnx")
onnx_output = session.run(None, {'input': dummy_input.numpy()})[0]

# 比较差异
diff = np.abs(torch_output - onnx_output)
print(f"最大误差: {diff.max():.6f}")
print(f"平均误差: {diff.mean():.6f}")
assert diff.max() < 1e-5, "ONNX导出误差过大！"
print("ONNX输出与PyTorch一致！")
```


> **Important:** ONNX导出时务必进行数值对比验证，确保导出的模型与原模型输出一致。常见误差来源包括：动态轴导致的算子执行路径不同、浮点精度累积差异、不支持的算子被自动替换等。


## 总结


- ONNX是开放的神经网络交换格式，实现训练框架与推理引擎的解耦
- PyTorch通过torch.onnx.export导出，支持动态轴配置和常量折叠优化
- ONNX Runtime提供多种硬件后端（CUDA、TensorRT、OpenVINO等）
- 图优化包括常量折叠、算子融合、布局转换等，可显著提升推理速度
- 导出后务必进行数值对比验证，确保输出与原始模型一致
- 生产环境中建议固定输入尺寸以获得最佳推理性能


<!-- Converted from: 01_ONNX与模型导出.html -->
