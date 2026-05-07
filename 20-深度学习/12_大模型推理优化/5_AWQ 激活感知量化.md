# 5_AWQ 激活感知量化

## 1. AWQ 核心思想

AWQ (Activation-aware Weight Quantization, Lin et al. 2024) 的核心洞察是：**不是所有权重同等重要**，1% 的"显著权重 (salient weights)"对模型质量影响最大。

```
AWQ vs GPTQ:

GPTQ: 逐层量化，用 Hessian 补偿误差
  - 需要计算 Hessian，速度较慢
  - 关注权重空间的误差最小化

AWQ: 保护显著权重，通过缩放变换
  - 不需要 Hessian，速度快
  - 利用激活分布识别重要权重
  - 通过 per-channel 缩放保护显著权重
```

### 显著权重现象

```
激活值分布:

  激活幅度
  │    ╭──╮
  │   ╱    ╲         大部分激活集中在 0 附近
  │  ╱      ╲
  │ ╱        ╲  ╭╮  ← 少量"显著通道"有大激活值
  │╱          ╲╱  ╲
  └──────────────────→ 通道索引

关键发现:
  - 小部分通道（~1%）的激活值远大于其他通道
  - 这些通道对应的权重对模型输出影响最大
  - 保护这些权重 = 保护模型质量
```

## 2. AWQ 算法

### 2.1 缩放变换

```python
"""
核心变换: per-channel 缩放

原始: Y = X @ W
变换: Y = (X / s) @ (W * s)   # 数学上等价

目标: 选择 s 使得量化误差最小
  - 对重要通道: s > 1 → W*s 更大 → 量化相对误差更小
  - 对普通通道: s < 1 → W*s 更小 → 接受更大相对误差
"""
```

### 2.2 搜索最优缩放系数

```python
import torch
import torch.nn as nn

class AWQQuantizer:
    """AWQ 激活感知权重量化"""

    def __init__(self, n_bits=4, group_size=128, search_grid_size=20):
        self.n_bits = n_bits
        self.group_size = group_size
        self.search_steps = search_grid_size

    def find_best_scale(self, W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        搜索最优缩放系数

        W: 权重矩阵 [out, in]
        X: 校准激活 [batch, seq, in]
        """
        # 1. 计算每个通道的激活重要性
        X_abs_mean = X.abs().mean(dim=(0, 1))  # [in_features]

        # 2. 候选缩放系数范围
        s_min = X_abs_mean.pow(-0.5).min()
        s_max = X_abs_mean.pow(-0.5).max()

        best_s = None
        best_error = float("inf")

        # 3. 网格搜索最优缩放系数
        for alpha in torch.linspace(0, 1, self.search_steps):
            # 缩放公式: s = activation_importance^alpha
            s = X_abs_mean.pow(alpha)
            s = s / s.mean()  # 归一化

            # 应用缩放
            W_scaled = W * s.unsqueeze(0)
            X_scaled = X / s.unsqueeze(0).unsqueeze(0)

            # 量化权重
            W_q = self.quantize_weight(W_scaled)

            # 计算重构误差
            Y_orig = X @ W.T
            Y_quant = X_scaled @ W_q.T * s.unsqueeze(0)
            error = (Y_orig - Y_quant).pow(2).mean().item()

            if error < best_error:
                best_error = error
                best_s = s

        return best_s

    def quantize_weight(self, W: torch.Tensor) -> torch.Tensor:
        """权重量化"""
        qmax = 2 ** (self.n_bits - 1) - 1

        # Per-group 量化
        out, inp = W.shape
        W_grouped = W.reshape(out, inp // self.group_size, self.group_size)

        abs_max = W_grouped.abs().max(dim=-1, keepdim=True).values
        scale = abs_max / qmax

        W_q = (W_grouped / scale).round().clamp(-qmax - 1, qmax)
        W_dequant = W_q * scale

        return W_dequant.reshape(out, inp)
```

### 2.3 完整模型量化流程

```python
class AWQModelQuantizer:
    """AWQ 全模型量化"""

    def __init__(self, model, n_bits=4, group_size=128):
        self.model = model
        self.n_bits = n_bits
        self.group_size = group_size
        self.awq = AWQQuantizer(n_bits, group_size)

    def collect_activation_stats(self, calibration_data):
        """收集每层的激活统计"""
        stats = {}

        def hook_fn(name):
            def hook(module, input, output):
                x = input[0].detach()
                if name not in stats:
                    stats[name] = x.cpu()
                else:
                    stats[name] = torch.cat([stats[name], x.cpu()], dim=0)
            return hook

        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            for data in calibration_data:
                self.model(data)

        for h in hooks:
            h.remove()

        return stats

    def quantize_model(self, calibration_data):
        """量化整个模型"""
        # 1. 收集激活统计
        print("收集激活统计...")
        activation_stats = self.collect_activation_stats(calibration_data)

        # 2. 逐层量化
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear) and name in activation_stats:
                print(f"量化: {name}")

                W = module.weight.data
                X = activation_stats[name]

                # 搜索最优缩放系数
                scale = self.awq.find_best_scale(W, X)

                # 应用缩放并量化
                W_scaled = W * scale.unsqueeze(0)
                W_q = self.awq.quantize_weight(W_scaled)

                # 替换权重
                module.weight.data = W_q / scale.unsqueeze(0)  # 反缩放到原始空间

        return self.model
```

## 3. AWQ 与硬件效率

```
AWQ 的硬件友好性:

1. 仅权重量化 (W4A16)
   - 权重存储为 INT4，计算时反量化为 FP16
   - 不需要 INT4 计算硬件支持
   - 兼容所有 GPU

2. Per-channel 缩放
   - 缩放系数在反量化时应用
   - 不增加额外计算开销

3. 与 GPTQ 的区别
   GPTQ: 需要在线反量化 + decompensation → 更复杂
   AWQ:  简单的 scale + dequantize → 更快
```

## 4. 实际使用

```python
# 使用 AWQ 官方库
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# 加载模型
model = AutoAWQForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 量化配置
quant_config = {
    "zero_point": True,
    "q_group_size": 128,
    "w_bit": 4,
    "version": "GEMM"  # 优化的 CUDA kernel
}

# 准备校准数据
calibration_data = [
    "your calibration text samples...",
    # 通常需要 50-200 条样本
]

# 执行量化
model.quantize(
    tokenizer,
    quant_config=quant_config,
    calib_data=calibration_data
)

# 保存
model.save_quantized("llama-2-7b-awq-4bit")
tokenizer.save_pretrained("llama-2-7b-awq-4bit")

# 加载量化模型
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("llama-2-7b-awq-4bit")
```

## 5. AWQ vs GPTQ 对比

```
┌──────────────┬────────────────┬────────────────┐
│    维度       │      AWQ       │     GPTQ       │
├──────────────┼────────────────┼────────────────┤
│ 核心方法      │ 缩放保护       │ Hessian 补偿   │
│ 需要激活统计  │ 是(统计量)     │ 是(Hessian)    │
│ 量化速度      │ 快             │ 慢             │
│ 推理速度      │ 快(简单反量化) │ 稍慢           │
│ INT4 质量     │ 优秀           │ 优秀           │
│ INT2 质量     │ 一般           │ 较好(QuIP#)    │
│ 硬件兼容性    │ 好             │ 需要特定 kernel│
│ 工具支持      │ vLLM, TGI     │ vLLM, TGI, llama.cpp │
└──────────────┴────────────────┴────────────────┘
```

## 6. AWQ 变体与改进

```python
# TEQ (Trainable Equivalent Transformation)
# 将缩放系数变为可训练参数
class TEQScaler(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_features))

    def forward(self, x, W):
        W_scaled = W * self.scale
        x_scaled = x / self.scale
        return x_scaled @ W_scaled.T

# 微调 scale 参数以最小化量化误差
def train_scale(model, calibration_data, n_epochs=10):
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad])

    for epoch in range(n_epochs):
        for data in calibration_data:
            output_q = model(data)  # 量化模型输出
            output_fp = model.forward_fp16(data)  # FP16 参考输出
            loss = (output_q - output_fp).pow(2).mean()
            loss.backward()
            optimizer.step()
```

## 总结

AWQ 利用**激活分布指导权重量化**，通过缩放变换保护显著权重，实现了**快速、高质量的 4-bit 量化**。相比 GPTQ，AWQ 更简单、更快，且推理时硬件效率更高。对于 4-bit 量化场景，AWQ 是目前的首选方案之一。
