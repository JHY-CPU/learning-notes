# TensorRT与推理优化 - 模型压缩与部署


## 1. NVIDIA TensorRT概述


TensorRT是NVIDIA提供的高性能深度学习推理优化器和运行时库，专门针对NVIDIA GPU进行深度优化。它可以将训练好的模型（来自PyTorch、TensorFlow等）优化为高度优化的推理引擎，显著提升推理速度。


### TensorRT核心优势


- **层融合（Layer Fusion）：**
   将多个连续层合并为一个计算核函数，减少GPU kernel launch开销和内存带宽
- **精度校准：**
   FP32→FP16/INT8精度转换，通过校准最小化精度损失
- **Kernel自动调优：**
   针对特定GPU架构选择最优的CUDA核函数实现
- **动态Tensor内存：**
   复用中间张量的显存，减少显存占用
- **多流并行：**
   利用CUDA Streams实现计算与数据传输的重叠


| 优化技术 | 原理 | 加速效果 |
| --- | --- | --- |
| Conv+BN+ReLU融合 | 3个kernel合并为1个 | 2-3x |
| FP16推理 | 利用Tensor Core半精度计算 | 1.5-3x |
| INT8推理 | 8位整数计算，吞吐量翻倍 | 2-4x |
| Kernel自动调优 | 选择最优算法和tile size | 1.2-2x |
| 显存优化 | 中间张量内存复用 | 减少50%显存 |


## 2. TensorRT工作流程


> **Example:** **模型部署完整流程：**
>
>
> PyTorch训练 → 导出ONNX → TensorRT解析优化 → 生成Engine → 部署推理服务


### 构建阶段（Build Phase）


1. 解析模型（ONNX Parser或TensorRT API直接构建）
2. 构建优化配置（精度模式、workspace大小、动态shape等）
3. 执行优化：层融合、精度校准、kernel选择
4. 序列化生成.plan引擎文件（可直接加载运行）


### 推理阶段（Inference Phase）


1. 反序列化加载Engine
2. 创建执行上下文（ExecutionContext）
3. 分配GPU显存，拷贝输入数据
4. 执行推理
5. 拷贝输出结果


```
import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# ============ 1. 构建TensorRT引擎 ============
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)

# 解析ONNX模型
with open("model.onnx", "rb") as f:
    if not parser.parse(f.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        raise ValueError("ONNX解析失败")

# 配置优化参数
config = builder.create_builder_config()
config.set_flag(trt.BuilderFlag.FP16)  # 启用FP16
config.max_workspace_size = 1 << 30    # 1GB workspace

# 构建并序列化引擎
engine = builder.build_engine(network, config)
with open("model.trt", "wb") as f:
    f.write(engine.serialize())

# ============ 2. 运行推理 ============
# 加载引擎
with open("model.trt", "rb") as f:
    engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())

context = engine.create_execution_context()

# 分配GPU内存
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(input_data.nbytes * 10)  # 预估输出大小

# 拷贝数据到GPU并推理
cuda.memcpy_htod(d_input, input_data)
context.execute_v2(bindings=[int(d_input), int(d_output)])

# 拷贝结果回CPU
output = np.empty([1, 1000], dtype=np.float32)
cuda.memcpy_dtoh(output, d_output)
print(f"推理结果形状: {output.shape}")
```


## 3. FP16/INT8精度推理


| 精度 | 显存占用 | 推理速度 | 精度损失 | 适用场景 |
| --- | --- | --- | --- | --- |
| FP32 | 基准 | 基准 | 无 | 训练、精度敏感任务 |
| FP16 | 50% | 1.5-3x | 极小（<0.1%） | 大多数推理场景 |
| INT8 | 25% | 2-4x | 较小（<1%） | 高吞吐量部署 |


### INT8校准流程


INT8量化需要通过校准数据集确定每层激活值的量化范围：


1. 准备代表性校准数据集（100-500个样本）
2. 运行校准数据，统计每层激活值的分布
3. 选择最优量化阈值（最大化信息熵或最小化KL散度）
4. 生成校准表，用于推理时的量化参数


> **Note:** FP16是TensorRT中最推荐的精度模式，因为大多数NVIDIA GPU（V100、A100、RTX系列）都有Tensor Core支持FP16加速，且几乎无精度损失。只有在极端吞吐量需求下才考虑INT8。


## 4. Triton Inference Server


Triton Inference Server是NVIDIA开源的模型服务框架，支持多种模型格式（TensorRT、ONNX、PyTorch、TensorFlow等），提供高性能的在线推理服务。


### 核心特性


- **多模型并发：**
   同时服务多个模型，共享GPU资源
- **动态批处理（Dynamic Batching）：**
   自动将多个请求合并为一个batch，提升GPU利用率
- **模型流水线（Ensemble）：**
   多模型组成推理流水线
- **模型版本管理：**
   支持同一模型的多版本部署和灰度切换
- **多框架支持：**
   TensorRT、ONNX、PyTorch、TensorFlow、OpenVINO
- **性能分析：**
   内置性能分析工具和指标导出


### 动态批处理


$$
无批处理：请求1→推理→响应1, 请求2→推理→响应2（2次推理）
                动态批处理：请求1+2→批量推理→响应1+2（1次推理，GPU利用率更高）
$$


| 参数 | 说明 | 推荐值 |
| --- | --- | --- |
| max_batch_size | 最大批次大小 | 根据显存和延迟要求设置（如32） |
| batching_timeout_microseconds | 等待凑批的超时时间 | 1000-5000（微秒） |
| preferred_batch_size | 优先批次大小 | [8, 16]（引擎针对这些大小优化） |


## 5. 云端 vs 边缘部署


| 对比维度 | 云端部署 | 边缘部署 |
| --- | --- | --- |
| 硬件 | NVIDIA A100/H100、多GPU集群 | Jetson、手机NPU、嵌入式GPU |
| 模型精度 | FP16/FP32为主 | INT8/INT4为主 |
| 优化工具 | TensorRT + Triton Server | TensorRT Lite、TFLite、NCNN、MNN |
| 关注指标 | 吞吐量（QPS）、延迟（P99） | 延迟、功耗、模型大小 |
| 更新方式 | 滚动更新、蓝绿部署 | OTA更新、A/B测试 |
| 典型场景 | 搜索引擎、推荐系统、大模型推理 | 自动驾驶、手机端AI、IoT设备 |


> **Important:** 边缘部署需要特别关注模型大小和功耗。NVIDIA Jetson系列提供了TensorRT支持，可以在嵌入式设备上实现高效推理。手机端可使用TensorFlow Lite、Core ML或NCNN。


## 6. 推理性能优化清单


1. **模型层面：**
   选择推理友好的模型架构（MobileNet、EfficientNet），避免动态控制流
2. **量化：**
   优先使用FP16，必要时使用INT8
3. **算子融合：**
   使用TensorRT等工具自动融合算子
4. **批处理：**
   使用动态批处理提升GPU利用率
5. **显存管理：**
   预分配显存池，避免频繁分配释放
6. **数据预处理：**
   使用GPU进行数据预处理（DALI），避免CPU瓶颈
7. **异步推理：**
   使用CUDA Stream实现计算与IO重叠
8. **多实例部署：**
   同一GPU部署多个模型实例
9. **缓存策略：**
   对相同输入缓存推理结果
10. **监控告警：**
   监控推理延迟、吞吐量、GPU利用率


## 总结


- TensorRT是NVIDIA的高性能推理引擎，通过层融合、精度优化、Kernel调优实现极致加速
- FP16是最推荐的推理精度，几乎无精度损失且有Tensor Core加速
- INT8推理需要校准，适合对吞吐量要求极高的场景
- Triton Inference Server提供完整的模型服务解决方案，支持动态批处理和多模型管理
- 云端部署关注吞吐量和延迟，边缘部署关注功耗和模型大小
- 完整的部署流程：训练→ONNX导出→TensorRT优化→Triton服务→监控运维


<!-- Converted from: 02_TensorRT与推理优化.html -->
