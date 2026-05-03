# 47_LLaMA：开源大模型的里程碑

## 核心概念

- **LLaMA (Large Language Model Meta AI)**：Meta 于 2023 年 2 月发布的大语言模型系列（7B、13B、33B、65B）。在更少的参数下实现了超越 GPT-3 的性能，被称为"开源大模型的里程碑"。
- **训练数据关键**：LLaMA 的核心理念是"数据质量比模型大小更重要"。使用 1.4T token（LLaMA-1）的训练数据，其中 67% 来自经过过滤的 CommonCrawl。
- **纯解码器架构 (Decoder-only)**：与 GPT 系列一样，LLaMA 使用因果掩码 Transformer 解码器，但引入了几项关键改进。
- **Pre-Normalization**：在注意力层和 FFN 层之前使用 RMSNorm 进行归一化（而非原始 Transformer 的 Post-Norm），提升了训练稳定性。
- **SwiGLU 激活函数**：在 FFN 中使用 SwiGLU（Swish-gated Linear Unit）替代 ReLU。SwiGLU 已被多项研究证明优于 ReLU 和 GELU。
- **RoPE 旋转位置编码**：使用 RoPE 替代绝对位置编码或可学习位置编码，提供更好的长度外推能力。
- **优化器设置**：使用 AdamW 优化器，配合 Cosine 学习率调度。独特之处在于对超参数 $\beta_2 = 0.95$（而非默认的 0.999），以及极小的 weight decay（0.1）。
- **LLaMA-2 和 LLaMA-3 的演进**：LLaMA-2 将训练数据增加到 2T token，上下文长度到 4096。LLaMA-3 进一步提升到 8B/70B 参数、8K 上下文，使用 GQA (Grouped Query Attention)。
- **开源生态的引爆点**：LLaMA 的开源引爆了整个大模型开源社区——基于 LLaMA 微调的模型（Alpaca、Vicuna、WizardLM）超过数百个，形成了"LLaMA 生态"。

## 数学推导

**RMSNorm**：
$$
\bar{x} = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}}
$$

$$
\text{RMSNorm}(x) = \bar{x} \odot \gamma
$$

与 LayerNorm 的区别：不进行中心化（不减均值），只进行缩放。计算更简单，实践中同样有效。

**SwiGLU**：
$$
\text{SwiGLU}(x, W, V, b) = \text{Swish}_\beta(xW) \odot (xV)
$$

$$
\text{Swish}_\beta(x) = x \cdot \sigma(\beta x)
$$

其中 $\beta$ 是学习参数（LLaMA 中固定 $\beta=1$）。

## 直观理解

- **LLaMA 像"精读名著的学生"**：很多大模型追求"读更多的书"（更大的参数量），而 LLaMA 追求"读更好的书"（更高质量的数据）。1.4T token 的训练数据花费大量功夫清洗和质量筛选，使得 13B 参数的 LLaMA 在多数基准上击败了 175B 的 GPT-3。
- **Pre-Norm 的好处**：在"吃饭前洗手"（子层处理前归一化）比"吃饭后洗手"（子层处理后归一化）对梯度流动更友好。Pre-Norm 是深层 Transformer 的训练小技巧，让 65B 的模型也可以稳定训练。
- **SwiGLU 像"有选择的信息门"**：传统 ReLU 是"负值一律杀死"（输出 0），SwiGLU 是"稍微负的值可以过但打折扣，正值保留并微调"。这种平滑的非线性让信息流动更自然。
- **开源里程碑的意义**：LLaMA 的开源就像 Android 对智能手机生态的影响——在它之前大模型是封闭的（GPT-3 API only），在它之后形成了繁荣的开源 LLM 生态。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    """RMS Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # x: (..., dim)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight

class SwiGLU(nn.Module):
    """SwiGLU 激活函数"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        # FFN 中三个权重矩阵：W_gate, W_up, W_down
        self.W_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.W_up = nn.Linear(dim, hidden_dim, bias=False)
        self.W_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        # SwiGLU(x) = Swish(W_gate x) * (W_up x)
        gate = F.silu(self.W_gate(x))  # SiLU = Swish
        up = self.W_up(x)
        return self.W_down(gate * up)

class LLaMABlock(nn.Module):
    """LLaMA Transformer 块"""
    def __init__(self, dim, n_heads, hidden_dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, bias=False)
        self.norm2 = RMSNorm(dim)
        self.swiglu = SwiGLU(dim, hidden_dim)

    def forward(self, x, mask=None):
        # Pre-Norm: 先归一化再注意力
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]
        # Pre-Norm: 先归一化再 FFN
        x = x + self.swiglu(self.norm2(x))
        return x

# LLaMA 架构参数示例
llama_configs = {
    "LLaMA-7B": {"dim": 4096, "n_layers": 32, "n_heads": 32, "hidden_dim": 11008},
    "LLaMA-13B": {"dim": 5120, "n_layers": 40, "n_heads": 40, "hidden_dim": 13824},
    "LLaMA-33B": {"dim": 6656, "n_layers": 60, "n_heads": 52, "hidden_dim": 17920},
    "LLaMA-65B": {"dim": 8192, "n_layers": 80, "n_heads": 64, "hidden_dim": 22016},
}

for name, cfg in llama_configs.items():
    params = cfg["n_layers"] * (4 * cfg["dim"] ** 2 + 3 * cfg["dim"] * cfg["hidden_dim"])
    print(f"{name}: dim={cfg['dim']}, layers={cfg['n_layers']}, heads={cfg['n_heads']}")
```

## 深度学习关联

- **开源 LLM 的分水岭**：LLaMA 的出现将大模型研究从"闭源 API 调用"推向"本地私有化部署"。其 Apache 2.0 许可证（LLaMA-2）也极具开放性。
- **微调生态的基础模型**：LLaMA 成为 Alpaca、Vicuna、WizardLM、Qwen、Yi 等众多微调模型的基础，证明了"预训练 + 指令微调"范式的可复制性。
- **架构改进的聚合**：LLaMA 汇集了 Pre-Norm、SwiGLU、RoPE、RMSNorm 等多项已被验证的训练技巧，展示了"工程优化"在大模型训练中的巨大价值。
