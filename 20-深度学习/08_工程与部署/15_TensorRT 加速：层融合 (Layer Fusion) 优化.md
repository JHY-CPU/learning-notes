# 16_TensorRT 加速：层融合 (Layer Fusion) 优化

## 核心概念

- **TensorRT**：NVIDIA 的高性能深度学习推理引擎，通过对计算图进行深度优化（层融合、精度校准、内存优化、内核自动调优），在 NVIDIA GPU 上实现极致的推理性能。最大优势是将模型的延迟降低到接近理论极限。
- **层融合 (Layer Fusion)**：TensorRT 的核心优化技术。通过将多个连续的计算层合并为一个核函数 (kernel)，减少显存访问和 kernel launch 开销。常见的融合模式包括：Conv + BN + ReLU 融合、矩阵乘 + Bias + GELU 融合等。
- **图优化 (Graph Optimization)**：除了层融合外，TensorRT 还会对计算图进行常量折叠 (Constant Folding)、死代码消除、算子消除等通用优化。所有优化都是自动的，无需手动干预。
- **INT8 校准 (INT8 Calibration)**：TensorRT 的 PTQ 支持通过校准数据集计算激活值的 KL 散度分布，自动选择最优的量化 scale 以最小化信息损失。支持逐张量 (per-tensor) 和逐通道 (per-channel) 量化。
- **动态形状 (Dynamic Shapes)**：支持推理时 batch size、输入尺寸动态变化的场景。在构建 TensorRT 引擎时需要指定优化范围（min/opt/max 三个维度），TensorRT 会在该范围内为所有可能的形状预编译优化内核。
- **Plugin (自定义插件)**：当模型包含 TensorRT 原生不支持的算子时，可以通过 Plugin API 编写自定义 CUDA 内核并注册到 TensorRT 中。Plugin 支持 FP32/FP16/INT8 多种精度。

## 数学推导

层融合的数学本质是将多个顺序操作的组合合并为一个算子，以减少中间张量的读写开销。

**Conv + BN + ReLU 融合**：

原始计算：
$$
\tilde{x} = \sum_{c} W_{c} * x_c + b \quad \text{(Conv)}
$$
$$
y = \gamma \cdot \frac{\tilde{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta \quad \text{(BN)}
$$
$$
z = \max(0, y) \quad \text{(ReLU)}
$$

融合后的单一算子：
$$
z = \max\left(0, \frac{\gamma}{\sqrt{\sigma^2 + \epsilon}} \cdot (W * x + b - \mu) + \beta\right)
$$
$$
= \max\left(0, \underbrace{\frac{\gamma W}{\sqrt{\sigma^2 + \epsilon}}}_{W_{\text{fused}}} * x + \underbrace{\frac{\gamma(b - \mu)}{\sqrt{\sigma^2 + \epsilon}} + \beta}_{b_{\text{fused}}}\right)
$$

这等价于一个带有融合偏置和 ReLU 的卷积层。避免了中间张量 $\tilde{x}$ 和 $y$ 的显存读写。

**Multi-Head Attention 融合**：在 Transformer 推理中，QKV 生成通常被融合为一个矩阵乘：

$$
[Q, K, V] = X \cdot W_{QKV}, \quad W_{QKV} = [W_Q, W_K, W_V] \in \mathbb{R}^{d \times 3d_k}
$$

相比分别计算三个投影，融合版本减少了 kernel launch 次数和显存访问。

## 直观理解

- **层融合 = 流水线工厂的工序合并**：想象一个汽车工厂有 4 道工序：喷漆 -> 烘干 -> 抛光 -> 质检。如果每道工序之间都需要把车运到另一个车间（读写显存），效率很低。层融合就是把这些工序放在同一个车间、同一条生产线上完成，中间不需要运输（不读写显存），效率自然大幅提升。
- **INT8 校准 = 为每个音量级别调整麦克风**：不同的输入音量需要不同的麦克风灵敏度。校准数据集就像是让模型对着各种音量的声音测试，TensorRT 自动为每一层选择合适的"麦克风灵敏度"（量化 scale），使得 INT8 量化后的精度损失最小。
- **最佳实践**：优先使用 FP16 精度（几乎无损），在精度满足要求的前提下使用 INT8；ONNX -> TensorRT 是标准路径，但也可以直接使用 PyTorch 的 `torch_tensorrt`。
- **常见陷阱**：动态形状引擎构建非常耗时（可能数小时），但构建后推理极快；务必在目标 GPU 型号上构建引擎，不同架构（Turing/Ampere/Hopper）的引擎不兼容。

## 代码示例

```python
import torch
import tensorrt as trt
import numpy as np
from typing import Optional

# ========== 1. ONNX -> TensorRT 引擎构建 ==========
def build_trt_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    int8: bool = False,
    int8_calibrator: Optional[trt.IInt8Calibrator] = None,
    dynamic_batch: bool = True,
    workspace_size: int = 4,  # GB
) -> trt.ICudaEngine:
    """将 ONNX 模型转换为 TensorRT 引擎"""
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 解析 ONNX 模型
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("ONNX 解析失败")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * 1 << 30)

    # 精度配置
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("启用 FP16")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if int8_calibrator:
            config.int8_calibrator = int8_calibrator
        print("启用 INT8")

    # 动态形状配置
    if dynamic_batch:
        profile = builder.create_optimization_profile()
        profile.set_shape("input", (1, 3, 224, 224), (4, 3, 224, 224), (16, 3, 224, 224))
        config.add_optimization_profile(profile)

    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("引擎构建失败")

    # 保存引擎
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    return serialized_engine

# ========== 2. TensorRT 推理 ==========
class TRTInference:
    """TensorRT 推理封装"""
    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()

        # 分配输入输出缓冲
        self.inputs, self.outputs, self.bindings = self._allocate_buffers()

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # 分配 GPU 内存
            buffer = torch.empty(size, dtype=torch.float32).cuda()
            binding_idx = self.engine.get_binding_index(binding)

            if self.engine.binding_is_input(binding):
                inputs.append(buffer)
            else:
                outputs.append(buffer)
            bindings.append(buffer.data_ptr())

        return inputs, outputs, bindings

    def infer(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """执行推理"""
        self.inputs[0].copy_(input_tensor.ravel())
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.cuda_stream
        )
        self.stream.synchronize()
        return self.outputs[0].clone()

# ========== 3. INT8 校准器 ==========
class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """基于熵的 INT8 校准器"""
    def __init__(self, dataloader, cache_file="calibration.cache"):
        super().__init__()
        self.dataloader = dataloader
        self.cache_file = cache_file
        self.data_iter = iter(dataloader)
        self.buffer = None

    def get_batch_size(self):
        return 4  # 与 dataloader 的 batch_size 一致

    def get_batch(self, names):
        try:
            data = next(self.data_iter)
            if self.buffer is None:
                self.buffer = torch.empty_like(data).cuda()
            self.buffer.copy_(data)
            return [self.buffer.data_ptr()]
        except StopIteration:
            return None

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)

# ========== 4. torch_tensorrt 方式（更 Pythonic）===========
def torch_trt_example():
    """使用 torch_tensorrt 直接优化 PyTorch 模型"""
    try:
        import torch_tensorrt
    except ImportError:
        print("需要安装 torch_tensorrt: pip install torch_tensorrt")
        return

    model = torch.nn.Sequential(
        torch.nn.Linear(1024, 4096),
        torch.nn.GELU(),
        torch.nn.Linear(4096, 1024),
    ).eval().cuda()

    # 编译模型
    trt_model = torch_tensorrt.compile(
        model,
        inputs=[torch.randn(1, 1024).cuda()],
        enabled_precisions={torch.float16},
        workspace_size=1 << 30,
        min_block_size=3,  # 至少融合 3 个算子
    )

    # 推理（与 PyTorch 代码完全一致）
    dummy = torch.randn(1, 1024).cuda().half()
    # output = trt_model(dummy)

# ========== 5. 性能对比 ==========
def benchmark():
    """对比原始 PyTorch 和 TensorRT 的推理速度"""
    import time

    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 64, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(64, 128, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d(1),
    ).eval().cuda()

    dummy = torch.randn(8, 3, 224, 224).cuda()

    # PyTorch FP32
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    pt_time = (time.time() - t0) / 100

    # 使用 torch.compile
    compiled = torch.compile(model)
    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            compiled(dummy)
    comp_time = (time.time() - t0) / 100

    print(f"PyTorch:  {pt_time*1000:.2f} ms")
    print(f"Compiled: {comp_time*1000:.2f} ms")
```

## 深度学习关联

- **云端推理服务的延迟优化**：TensorRT 是为生产级推理服务降低延迟和提升吞吐量的核心工具。在 Triton Inference Server 中，TensorRT 作为 backend 之一自动为模型提供层融合和 INT8 优化。典型的延迟降低幅度为 FP32 的 3-5 倍，INT8 的 10-20 倍。
- **A/B 测试中的精度验证**：在 MLOps 的 A/B 测试中，TensorRT 优化后的模型需要对照原始 PyTorch 模型进行精度验证。常见的做法是在 CI 流水线中运行一个包含数千个样本的精度测试集，确保 TensorRT 引擎的精度损失（尤其是 INT8 量化后）在可接受范围内（通常 < 0.5%）。
- **持续部署中的引擎缓存**：TensorRT 引擎构建耗时较长（大型模型可能需要数小时），因此在持续部署流水线中通常使用构建缓存。当模型权重未变化时直接复用缓存的 TensorRT 引擎，仅在模型更新时重新构建。引擎文件通常与模型版本一起存储在对象存储（S3/GCS）中。
