# 13_量化技术：PTQ (训练后量化) 基础

## 核心概念

- **模型量化 (Quantization)**：将模型的权重和激活值从 FP32 降低到更低比特（INT8、INT4、甚至二值化），以减少模型存储大小和推理延迟。量化在移动端、边缘设备和云端推理优化中广泛使用。
- **PTQ (Post-Training Quantization)**：训练完成后直接对模型进行量化，无需重新训练或微调。通过在校准数据集上统计激活值分布来确定量化参数（scale 和 zero_point），是最简单快速的量化方式。
- **对称量化 vs 非对称量化**：对称量化将数值范围对称映射到 [-127, 127]（zero_point=0），适合权重分布对称的场景；非对称量化允许 zero_point 非零，可以适应任意分布，通常用于激活值量化。
- **逐层量化 vs 逐通道量化**：逐层量化对整层使用相同的 scale/zero_point；逐通道量化对每个输出通道独立计算量化参数，精度更好但实现更复杂。权重量化常用逐通道，激活量化常用逐层。
- **校准 (Calibration)**：PTQ 中使用少量代表性数据（通常 100-500 个样本）前向传播模型，收集每层的激活值统计分布（min/max、百分位数、KL 散度等），以此确定量化参数。
- **量化粒度**：除了逐层/逐通道外，还有按张量（per-tensor）和按组（per-group, group_size=32/64/128）的量化粒度。更细的粒度带来更好的精度但增加存储开销。group 量化在 LLM 量化中非常常见。

## 数学推导

量化将一个连续的浮点数范围映射到离散的整数值。

**非对称量化映射**：

$$
q = \text{round}\left(\frac{r - r_{\min}}{s}\right) + \text{zero\_point}
$$
$$
r = (q - \text{zero\_point}) \times s
$$

其中 $r$ 是浮点值，$q$ 是量化后的整数值，$s$ 是缩放因子（scale），zero_point 是零点偏移：

$$
s = \frac{r_{\max} - r_{\min}}{2^n - 1}, \quad n \text{ 为比特数}
$$

**对称量化映射**（$r_{\min} = -r_{\max}$，zero_point = 0）：

$$
q = \text{round}\left(\frac{r}{s}\right), \quad s = \frac{\max(|r_{\max}|, |r_{\min}|)}{2^{n-1} - 1}
$$

INT8 量化的量化误差分析。量化误差主要来自 rounding：

$$
\text{MSE}_{\text{quant}} = \mathbb{E}\left[(r - \tilde{r})^2\right] = \sum_{i=1}^N \int_{b_i}^{b_{i+1}} (r - q_i)^2 p(r) dr
$$

其中 $b_i$ 是量化边界，$q_i$ 是量化电平，$p(r)$ 是权重/激活值的概率分布。

**矩阵乘法的 INT8 量化**：对于 $Y = WX + B$，量化版本为：

$$
q_Y = \text{clamp}\left( \text{round}\left( \frac{s_W s_X}{s_Y} (q_W - z_W) (q_X - z_X) + \frac{s_B}{s_Y}(q_B - z_B) + z_Y \right), -128, 127 \right)
$$

这涉及浮点数的乘加操作，因此 INT8 推理通常使用"量化-反量化"流水线或使用 INT8 张量核心（Tensor Core）直接计算。

## 直观理解

- **量化 = 照片压缩**：FP32 好比是 32 位色深的 RAW 照片——信息丰富、文件大；INT8 好比是 256 色调色板的 JPEG——文件小、速度快，但色彩细节略有损失。关键是要找到"人眼看不出来"（模型精度不明显下降）的压缩率。
- **PTQ vs QAT**：PTQ 是在照片拍完后压缩成 JPEG（快速但可能画质下降）；QAT 是在拍照时就考虑到压缩需求，让相机适应该压缩算法（稍慢但效果更好）。
- **最佳实践**：从 PTQ (INT8) 开始尝试，如果精度满足需求就直接使用；如果精度损失大，再尝试 QAT 或 INT4/FP8 量化。
- **常见陷阱**：校准数据集必须来自真实的训练/推理分布。如果校准数据与真实数据分布有偏移，量化参数会不准确，导致推理精度异常下降。

## 代码示例

```python
import torch
import torch.nn as nn
from torch.quantization import (
    quantize_dynamic,
    quantize_qat,
    prepare,
    convert,
    get_default_qconfig,
    QConfig,
)

# ========== 1. 动态量化 (Dynamic Quantization) ==========
# 只量化权重，激活值在推理时动态量化
model = nn.Sequential(
    nn.Linear(768, 3072),
    nn.GELU(),
    nn.Linear(3072, 768),
)

# 一行代码完成动态量化
quantized_model = quantize_dynamic(
    model,
    {nn.Linear},  # 只量化 Linear 层
    dtype=torch.qint8
)

print(f"原始模型大小: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.2f} MB")
print(f"量化后模型大小: {sum(p.numel() for p in quantized_model.parameters()) / 1024**2:.2f} MB")

# 推理方式不变
dummy_input = torch.randn(1, 768)
with torch.no_grad():
    output = quantized_model(dummy_input)

# ========== 2. 静态量化 (Static Quantization) ==========
# 需要校准数据来确定激活值的量化参数

# 2.1 准备量化模型
class QuantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.conv = nn.Conv2d(3, 64, 3)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(64*222*222, 10)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

model_fp32 = QuantModel().eval()
model_fp32.qconfig = get_default_qconfig("x86")  # 或 "fbgemm"

# 准备量化（插入 Observer）
prepared_model = prepare(model_fp32, inplace=False)

# 2.2 校准：用少量数据观察激活值分布
def calibrate(model, calib_loader, num_batches=10):
    model.eval()
    with torch.no_grad():
        for i, (data, _) in enumerate(calib_loader):
            if i >= num_batches:
                break
            model(data)

# calibrate(prepared_model, calibration_loader)

# 2.3 转换为量化模型
# quantized_model = convert(prepared_model)

# ========== 3. INT4 量化（使用 bitsandbytes 或 GPTQ）===========
# 对于 LLM 常用的 4-bit 量化
# pip install bitsandbytes
# import bitsandbytes as bnb

# 使用 bnb 的 4-bit 量化：
# from transformers import BitsAndBytesConfig
# config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )

# ========== 4. 手动实现对称量化 ==========
def symmetric_quantize(tensor, num_bits=8):
    """对称量化"""
    max_val = tensor.abs().max()
    q_max = 2 ** (num_bits - 1) - 1  # 127 for INT8
    scale = max_val / q_max

    # 量化
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, -q_max - 1, q_max)
    # 反量化评估误差
    deq_tensor = q_tensor * scale
    mse = ((tensor - deq_tensor) ** 2).mean().item()
    return q_tensor.to(torch.int8), scale, mse

# 测试
w = torch.randn(1024, 1024)  # FP32 权重
q_w, scale, mse = symmetric_quantize(w)
print(f"INT8 量化 MSE: {mse:.6f}")

# ========== 5. 量化模型大小和速度评估 ==========
def evaluate_quantization(model, quantized_model, input_shape=(1, 3, 224, 224)):
    """对比量化前后的模型大小和推理速度"""
    import time

    # 模型大小
    def model_size(m):
        return sum(p.numel() * p.element_size() for p in m.parameters()) / 1024**2

    fp32_size = model_size(model)
    quant_size = model_size(quantized_model)

    # 推理速度
    dummy = torch.randn(*input_shape)
    model.eval()
    quantized_model.eval()

    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            model(dummy)
    fp32_time = (time.time() - t0) / 100

    t0 = time.time()
    with torch.no_grad():
        for _ in range(100):
            quantized_model(dummy)
    quant_time = (time.time() - t0) / 100

    print(f"FP32: {fp32_size:.2f} MB, {fp32_time*1000:.2f} ms")
    print(f"INT8: {quant_size:.2f} MB, {quant_time*1000:.2f} ms")
    print(f"加速比: {fp32_time/quant_time:.2f}x")
```

## 深度学习关联

- **云端推理服务的成本优化**：在 Triton/TorchServe 中部署量化模型可使每 GPU 吞吐量提高 2-4 倍，从而在相同 QPS 要求下减少 GPU 数量，显著降低云端推理成本。这直接转化为 MLOps 中的 ROI 提升。
- **边缘设备部署标准流程**：PTQ 是移动端/边缘设备部署的标准第一步。通常流程为：PyTorch 训练(F32) -> ONNX 导出 -> ONNX Runtime 量化/ TensorRT INT8 校准 -> 边缘设备部署。每个步骤都需要在 CI/CD 流水线中自动验证精度。
- **LLM 量化与模型服务**：对于 7B/13B/70B 等大语言模型，INT4/INT8 量化是实际部署的关键前提。GPTQ、AWQ、GGUF 等量化算法的 PTQ 变体使得 70B 模型从需要 4 张 A100 压缩到仅需 1 张。在 MLflow 中，量化配置（bits、group_size、quant_type）应与模型版本一起记录。
