# 3_量化基础：FP16/BF16/INT8/INT4

## 1. 数据格式概览

量化的核心是**用更少的比特位表示模型参数**，从而减少内存占用和加速计算。

```
数据格式精度与范围对比:

格式    │ 位数 │ 符号 │ 指数 │ 尾数 │     范围       │ 精度
────────┼──────┼──────┼──────┼──────┼────────────────┼────────
FP32    │  32  │  1   │  8   │  23  │ ±3.4×10³⁸     │ 高
FP16    │  16  │  1   │  5   │  10  │ ±65504         │ 中
BF16    │  16  │  1   │  8   │  7   │ ±3.4×10³⁸     │ 低
INT8    │   8  │  1   │  -   │  7   │ [-128, 127]    │ 离散
INT4    │   4  │  1   │  -   │  3   │ [-8, 7]        │ 离散
NF4     │   4  │  1   │  -   │  3   │ 非均匀分布     │ 离散

内存占用对比 (1B 参数):
  FP32: 4 GB
  FP16/BF16: 2 GB
  INT8: 1 GB
  INT4: 0.5 GB
```

### FP16 vs BF16

```
FP16 (IEEE 754 Half Precision):
  1 bit sign | 5 bits exponent | 10 bits mantissa
  优点: 精度较高
  缺点: 范围小，训练时容易溢出

BF16 (Brain Float 16):
  1 bit sign | 8 bits exponent | 7 bits mantissa
  优点: 范围与 FP32 相同，不容易溢出
  缺点: 精度较低
  结论: 训练首选 BF16，推理两者差异不大
```

## 2. 量化基本原理

### 2.1 线性量化

```python
import numpy as np

def linear_quantize(tensor, n_bits=8, scheme="symmetric"):
    """
    线性量化: 将浮点数映射到整数

    量化公式 (对称):
      scale = max(|tensor|) / (2^(n-1) - 1)
      q = round(tensor / scale)
      tensor_hat = q * scale

    量化公式 (非对称):
      scale = (max - min) / (2^n - 1)
      zero_point = round(-min / scale)
      q = round(tensor / scale) + zero_point
      tensor_hat = (q - zero_point) * scale
    """
    qmin = -(2 ** (n_bits - 1))      # -128 for INT8
    qmax = 2 ** (n_bits - 1) - 1     #  127 for INT8

    if scheme == "symmetric":
        abs_max = np.abs(tensor).max()
        scale = abs_max / qmax
        zero_point = 0
    else:  # asymmetric
        scale = (tensor.max() - tensor.min()) / (qmax - qmin)
        zero_point = np.round(-tensor.min() / scale).astype(np.int8)

    # 量化
    q = np.round(tensor / scale + zero_point).clip(qmin, qmax).astype(np.int8)

    # 反量化 (用于验证)
    dequantized = (q.astype(np.float32) - zero_point) * scale

    # 量化误差
    error = np.abs(tensor - dequantized).mean()

    return {
        "quantized": q,
        "scale": scale,
        "zero_point": zero_point,
        "dequantized": dequantized,
        "error": error,
        "compression_ratio": 32 / n_bits
    }

# 示例
tensor = np.random.randn(1000).astype(np.float32) * 10
result = linear_quantize(tensor, n_bits=8, scheme="symmetric")
print(f"量化误差: {result['error']:.6f}")
print(f"压缩比: {result['compression_ratio']}x")
```

### 2.2 量化粒度

```python
"""
量化粒度决定了 scale/zero_point 的计算范围:

Per-Tensor:  整个张量共享一个 scale
  scale = max(|W|) / qmax
  最简单，精度最差

Per-Channel: 每个输出通道独立 scale
  scale_c = max(|W_c|) / qmax  (c = 1..C_out)
  精度好，开销适中，最常用

Per-Group:   每 g 个元素一组
  scale_g = max(|W_g|) / qmax  (g = 1..G)
  精度最好，开销较大
  AWQ/GPTQ 常用 group_size=128
"""

class PerChannelQuantizer:
    """Per-Channel 量化"""

    def quantize(self, weight: np.ndarray, n_bits=8):
        # weight: [out_features, in_features]
        qmax = 2 ** (n_bits - 1) - 1

        # 每个输出通道一个 scale
        abs_max = np.abs(weight, axis=1, keepdims=True)
        scale = abs_max / qmax

        q = np.round(weight / scale).clip(-qmax - 1, qmax).astype(np.int8)
        return q, scale

class PerGroupQuantizer:
    """Per-Group 量化"""

    def quantize(self, weight: np.ndarray, n_bits=8, group_size=128):
        out_features, in_features = weight.shape
        assert in_features % group_size == 0

        qmax = 2 ** (n_bits - 1) - 1
        num_groups = in_features // group_size

        # reshape 为组
        w_grouped = weight.reshape(out_features, num_groups, group_size)

        # 每组计算 scale
        abs_max = np.abs(w_grouped).max(axis=2, keepdims=True)
        scale = abs_max / qmax

        q = np.round(w_grouped / scale).clip(-qmax - 1, qmax).astype(np.int8)
        return q.reshape(weight.shape), scale
```

## 3. 量化误差分析

```python
def quantization_error_analysis(tensor, bit_widths=[16, 8, 4, 2]):
    """分析不同位宽的量化误差"""
    results = {}

    for n_bits in bit_widths:
        result = linear_quantize(tensor, n_bits=n_bits)
        reconstructed = result["dequantized"]

        # 多种误差指标
        mse = np.mean((tensor - reconstructed) ** 2)
        mae = np.mean(np.abs(tensor - reconstructed))
        max_error = np.max(np.abs(tensor - reconstructed))
        snr = 10 * np.log10(np.var(tensor) / mse)  # 信噪比

        results[n_bits] = {
            "MSE": mse,
            "MAE": mae,
            "Max_Error": max_error,
            "SNR_dB": snr,
            "memory_reduction": 32 / n_bits
        }

    return results

# 典型结果
"""
位宽 │  MSE      │  SNR (dB)  │  内存节省
─────┼───────────┼────────────┼──────────
16   │  1e-7     │  70        │  2x
 8   │  1e-5     │  50        │  4x
 4   │  1e-3     │  30        │  8x
 2   │  1e-1     │  10        │  16x

结论:
- INT8: 精度损失极小，几乎无损
- INT4: 轻微损失，大多数任务可接受
- INT2: 显著损失，需要特殊技术弥补
"""
```

## 4. NF4 (Normal Float 4-bit)

```
NF4 是 QLoRA 提出的最优 4-bit 数据类型

思想: 假设权重服从正态分布，
设计非均匀量化级别使得每个 bin 的期望数据量相等

标准 INT4: [-8, -7, ..., 0, ..., 7] (均匀分布)
NF4:       [-7.5, -6.0, -4.5, -3.0, -2.0, -1.0, -0.5, 0,
             0.5, 1.0, 2.0, 3.0, 4.5, 6.0, 7.5] (非均匀)

NF4 的优势: 对正态分布的权重，信息论上最优
```

```python
# NF4 量化级别 (信息论最优)
NF4_LEVELS = [
    -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848,
    -0.0911, 0.0, 0.0796, 0.1609, 0.2586, 0.3661,
    0.4984, 0.6701, 0.9197, 1.0
]

def nf4_quantize(tensor):
    """NF4 量化"""
    # 归一化到 [-1, 1]
    absmax = tensor.abs().max()
    normalized = tensor / absmax

    # 找到最近的 NF4 级别
    levels = torch.tensor(NF4_LEVELS, device=tensor.device)
    indices = torch.abs(normalized.unsqueeze(-1) - levels).argmin(dim=-1)

    return indices, absmax

def nf4_dequantize(indices, absmax):
    """NF4 反量化"""
    levels = torch.tensor(NF4_LEVELS, device=indices.device)
    return levels[indices] * absmax
```

## 5. 量化对模型质量的影响

```
量化误差的影响因素:

1. 模型大小
   大模型对量化更鲁棒（参数冗余多）
   70B INT4 ≈ 13B FP16 质量

2. 量化粒度
   Per-Channel > Per-Group > Per-Tensor

3. 校准数据集
   权重量化: 不需要校准
   激活量化: 需要校准数据确定动态范围

4. 异常值处理
   激活中的异常值是 INT8 量化的主要挑战
   SmoothQuant 通过变换解决
```

## 6. 实践指南

```python
"""
量化选择指南:

场景                          推荐方案
─────────────────────────────── ──────────────
有 GPU，追求速度               INT8 (W8A8)
有 GPU，追求压缩比             INT4 (W4A16)
仅有 CPU                       INT4 GGUF
需要最高质量                   BF16 / FP16
微调 + 量化                    QLoRA (NF4)
边缘设备                       INT4 + 蒸馏

精度-速度权衡:
  FP16 → INT8: ~1.5-2x 加速，<1% 精度损失
  FP16 → INT4: ~2-3x 加速，1-3% 精度损失
  FP16 → INT2: ~4x 加速，>5% 精度损失
"""
```

## 总结

量化是大模型推理优化最有效的手段之一。**INT8 近乎无损**，是通用推荐方案；**INT4 通过 GPTQ/AWQ 等技术实现质量与压缩的平衡**；**NF4 是针对正态分布权重的理论最优 4-bit 格式**。选择量化方案需综合考虑硬件、延迟要求和质量容忍度。
