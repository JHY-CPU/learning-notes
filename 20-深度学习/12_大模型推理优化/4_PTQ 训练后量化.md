# 4_PTQ 训练后量化

## 1. PTQ 概述

训练后量化 (Post-Training Quantization, PTQ) 是在**不重新训练模型**的情况下，直接对预训练模型进行量化的方法。

```
PTQ vs QAT (量化感知训练):

PTQ: 预训练模型 → 校准 → 量化模型
     优点: 简单快速，几小时内完成
     缺点: 低位宽(4bit)时精度损失较大

QAT: 预训练模型 → 量化感知微调 → 量化模型
     优点: 精度更好
     缺点: 需要训练资源，耗时较长

主流 PTQ 方法:
  - GPTQ: 逐层量化 + 最优脑损伤补偿
  - SmoothQuant: 激活-权重联合变换
  - AWQ: 激活感知权重保护
  - QuIP#: 非凸优化 + lattice 量化
```

## 2. GPTQ 算法

### 2.1 核心思想

GPTQ (Frantar et al., 2023) 基于**最优脑量化 (OBQ)** 算法，逐层量化权重，同时用 Hessian 信息补偿量化误差。

```
核心直觉:
  不是简单地 round(W / scale) * scale
  而是考虑权重之间的相关性，
  量化一个权重后，调整其他权重来补偿误差

数学原理:
  量化误差 = (W - W_quantized)²
  最优补偿: δ = (W_row - W_quantized_row) / H_diag
  其中 H 是 Hessian 矩阵的对角近似
```

### 2.2 GPTQ 算法实现

```python
import torch
import numpy as np

class GPTQQuantizer:
    """GPTQ 逐层量化"""

    def __init__(self, n_bits=4, group_size=128, percdamp=0.01):
        self.n_bits = n_bits
        self.group_size = group_size
        self.percdamp = percdamp  # Hessian 对角项的阻尼系数

    def quantize_layer(self, W: torch.Tensor, H_inv: torch.Tensor) -> dict:
        """
        量化单层权重矩阵

        W: 权重矩阵 [out_features, in_features]
        H_inv: Hessian 逆矩阵的对角近似 [in_features, in_features]
        """
        rows, cols = W.shape
        W_copy = W.clone()

        # 存储量化结果
        Q = torch.zeros_like(W)
        scales = torch.zeros(rows, cols // self.group_size)
        zeros = torch.zeros(rows, cols // self.group_size)

        # 按行处理
        for i in range(rows):
            w = W_copy[i]
            h = H_inv.diag()

            # 分组量化
            for g_start in range(0, cols, self.group_size):
                g_end = min(g_start + self.group_size, cols)
                group_w = w[g_start:g_end]
                group_h = h[g_start:g_end]

                # 计算量化参数
                qmin = -(2 ** (self.n_bits - 1))
                qmax = 2 ** (self.n_bits - 1) - 1

                abs_max = group_w.abs().max()
                scale = abs_max / qmax

                # 量化
                q_w = (group_w / scale).round().clamp(qmin, qmax)

                # 计算量化误差
                error = (group_w - q_w * scale)

                # 补偿: 将误差分散到未量化的列
                # 这是 GPTQ 的核心 — 利用 Hessian 信息
                if g_end < cols:
                    compensation = error / (group_h.mean() + self.percdamp)
                    W_copy[i, g_end:] -= compensation.mean()

                # 存储结果
                Q[i, g_start:g_end] = q_w
                scales[i, g_start // self.group_size] = scale

        return {
            "quantized": Q.to(torch.int8),
            "scales": scales,
            "original_shape": W.shape,
            "n_bits": self.n_bits,
            "group_size": self.group_size
        }
```

### 2.3 Hessian 近似

```python
def compute_hessian_diagonal(W, calibration_data, layer):
    """用校准数据近似 Hessian 对角项"""
    # 前向传播获取激活值
    activations = []
    for x in calibration_data:
        with torch.no_grad():
            act = layer.input_layernorm(x)  # 获取层输入
            activations.append(act)

    # Hessian 对角近似 ≈ 2 * E[X * X.T]
    H_diag = torch.zeros(W.shape[1])
    for act in activations:
        H_diag += (act ** 2).mean(dim=0)

    H_diag /= len(activations)
    H_diag += 0.01  # 阻尼项

    return H_diag
```

## 3. SmoothQuant

### 3.1 核心思想

SmoothQuant 解决了**激活量化**中的异常值问题。

```
问题: 激活中存在异常值 (outliers)，导致量化范围被撑大

  权重 W: 分布均匀 → 容易量化
  激活 X: 有异常值 → 量化误差大

SmoothQuant 的解决:
  将量化难度从激活"迁移"到权重

  Y = X @ W = (X / s) @ (W * s)

  选择 s 使得 X/s 更均匀（消除异常值影响）
  代价是 W*s 稍微不均匀，但权重本身就是静态的，可以精细量化
```

```python
class SmoothQuant:
    """SmoothQuant: 激活-权重平滑量化"""

    def __init__(self, alpha=0.5):
        self.alpha = alpha  # 平滑系数

    def smooth_layer(self, W: torch.Tensor, X_stats: dict) -> tuple:
        """
        W: 权重矩阵 [out, in]
        X_stats: 激活统计 {"abs_max": [in_features], ...}
        """
        # 计算平滑系数
        # s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha)
        x_scale = X_stats["abs_max"] ** self.alpha
        w_scale = W.abs().max(dim=0).values ** (1 - self.alpha)

        s = x_scale / w_scale
        s = s.clamp(min=1e-5)  # 防止除零

        # 变换: W' = W * s, 后续 X' = X / s 合并到上一层
        W_smoothed = W * s

        return W_smoothed, s

    def calibrate(self, model, calibration_data):
        """使用校准数据收集激活统计信息"""
        activation_stats = {}

        def hook_fn(name):
            def hook(module, input, output):
                x = input[0].detach()
                if name not in activation_stats:
                    activation_stats[name] = {"abs_max": x.abs().max(dim=0).values}
                else:
                    activation_stats[name]["abs_max"] = torch.max(
                        activation_stats[name]["abs_max"],
                        x.abs().max(dim=0).values
                    )
            return hook

        # 注册 hook 收集激活统计
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                h = module.register_forward_hook(hook_fn(name))
                hooks.append(h)

        # 校准
        with torch.no_grad():
            for data in calibration_data:
                model(data)

        # 移除 hooks
        for h in hooks:
            h.remove()

        return activation_stats
```

## 4. 逐层量化的实现细节

```python
class LayerByLayerQuantizer:
    """完整的逐层量化流程"""

    def __init__(self, model, n_bits=4, method="gptq"):
        self.model = model
        self.n_bits = n_bits
        self.method = method

    def quantize_model(self, calibration_data) -> dict:
        """量化整个模型"""
        quantized_layers = {}
        layer_names = []

        # 收集需要量化的层
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                layer_names.append(name)

        # 逐层量化
        for i, name in enumerate(layer_names):
            print(f"量化层 {i+1}/{len(layer_names)}: {name}")

            layer = self.get_layer_by_name(name)
            W = layer.weight.data

            if self.method == "gptq":
                # 获取 Hessian 信息
                H_diag = self.get_hessian(name, calibration_data)
                quantizer = GPTQQuantizer(n_bits=self.n_bits)
                result = quantizer.quantize_layer(W, H_diag)

            elif self.method == "smoothquant":
                # 收集激活统计
                act_stats = self.get_activation_stats(name, calibration_data)
                sq = SmoothQuant(alpha=0.5)
                W_smoothed, s = sq.smooth_layer(W, act_stats)
                result = self.simple_quantize(W_smoothed)

            quantized_layers[name] = result

            # 用量化后的权重替换原始权重（供后续层校准）
            self.replace_weight(name, result["dequantized"])

        return quantized_layers
```

## 5. 实际使用：GPTQ 量化工具

```python
# 使用 auto-gptq 库
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,
    group_size=128,
    desc_act=True,     # 激活排序
    sym=True,          # 对称量化
    damp_percent=0.01,
)

# 加载模型
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config,
)

# 准备校准数据
calibration_data = [...]  # 128-512 条样本

# 执行量化
model.quantize(calibration_data)

# 保存量化模型
model.save_quantized("llama-2-7b-gptq-4bit")

# 加载量化模型
quantized_model = AutoGPTQForCausalLM.from_quantized(
    "llama-2-7b-gptq-4bit",
    device_map="auto"
)
```

## 6. PTQ 方法对比

```
┌────────────┬────────┬────────┬─────────┬──────────────┐
│   方法      │ 位宽   │ 需要激活│ 速度    │ 质量          │
├────────────┼────────┼────────┼─────────┼──────────────┤
│ GPTQ       │ 4-bit  │ Hessian│ 慢(小时)│ 优秀          │
│ AWQ        │ 4-bit  │ 统计量 │ 快      │ 优秀          │
│ SmoothQuant│ 8-bit  │ 统计量 │ 快      │ 优秀(W8A8)   │
│ QuIP#      │ 2-bit  │ Hessian│ 很慢    │ 良好(2bit)    │
│ HQQ        │ 4-bit  │ 无     │ 最快    │ 良好          │
│ GGUF       │ 2-8bit │ 无     │ 快      │ 良好(CPU)     │
└────────────┴────────┴────────┴─────────┴──────────────┘
```

## 总结

GPTQ 是目前最成熟的**权重量化 PTQ 方法**，核心是利用 Hessian 信息进行误差补偿。SmoothQuant 解决了激活量化的异常值问题，使得 **W8A8 量化**成为可能。选择 PTQ 方法需综合考虑：量化位宽需求、校准数据可用性、以及目标硬件平台。
