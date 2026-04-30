# 31_RoPE 旋转位置编码原理与实现

## 核心概念
- **RoPE (Rotary Position Embedding)**：由苏剑林 (2021) 提出，通过旋转矩阵对词向量施加位置信息，是当前最流行的相对位置编码方案之一（LLaMA、Qwen、GLM 等均使用）。
- **核心思想**：将位置编码转化为对 Query 和 Key 向量的旋转操作。两个向量的点积结果会自动编码它们之间的相对位置信息。
- **相对位置编码**：RoPE 使注意力分数只依赖于 Query 和 Key 之间的相对位置差 $(m-n)$，而非绝对位置 $m$ 和 $n$，这有助于模型泛化到更长的序列。
- **旋转矩阵实现**：将 $d$ 维向量分成 $d/2$ 个二维子空间，在每个子空间中对 $(q_{2i}, q_{2i+1})$ 应用 $m\theta_i$ 角度的旋转。
- **长度外推能力**：RoPE 理论支持任意长度的外推，因为旋转操作本身不受序列长度限制。但在实践中，外推效果受限于训练时的位置分布。
- **无需额外参数**：与可学习位置编码不同，RoPE 不引入新的可学习参数，完全基于数学变换。
- **线性注意力兼容**：RoPE 的旋转操作与线性注意力机制兼容，可以应用于 Transformer 的变体。

## 数学推导
RoPE 的核心公式：
$$
\text{RoPE}(x_m, m) = R_m \cdot x_m
$$

其中 $R_m$ 是块对角旋转矩阵：
$$
R_m = \begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & \cdots & 0 \\
\sin m\theta_0 & \cos m\theta_0 & 0 & \cdots & 0 \\
0 & 0 & \cos m\theta_1 & \cdots & 0 \\
0 & 0 & \sin m\theta_1 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & \cos m\theta_{d/2-1} & -\sin m\theta_{d/2-1} \\
0 & 0 & 0 & \cdots & \sin m\theta_{d/2-1} & \cos m\theta_{d/2-1}
\end{pmatrix}
$$

频率 $\theta_i = 10000^{-2i/d}$，与正弦-余弦位置编码相同。

**关键性质**：旋转后的 Query 和 Key 的点积包含相对位置信息：
$$
(R_m q)^\top (R_n k) = q^\top R_{n-m} k
$$

这意味着注意力分数 $q_m^\top k_n$ 只依赖于相对位置 $(m-n)$，而不是绝对位置 $m, n$。

高效的实现方式（无需完整矩阵乘法）：
$$
\text{RoPE}(x_m)_i = \begin{cases}
x_m[i] \cos m\theta_j - x_m[i+1] \sin m\theta_j & \text{if } i \text{ even} \\
x_m[i-1] \sin m\theta_j + x_m[i] \cos m\theta_j & \text{if } i \text{ odd}
\end{cases}
$$

其中 $j = \lfloor i/2 \rfloor$。

## 直观理解
- **RoPE 像旋转时钟指针**：想象一个时钟，位置 $m$ 就是时间。旋转 Query 向量 $\theta_i$ 角度就像让时钟指针转动到对应时间的位置。两个时间 $m$ 和 $n$ 的指针夹角 $(m-n)\theta_i$ 自然地编码了时间差。
- **分组旋转**：RoPE 将高维向量分成 $d/2$ 个二维组，每个组以不同的速度旋转。低维组旋转慢（相当于时针），高维组旋转快（相当于秒针）——不同频率的组合唯一地标识了每个位置。
- **旋转而非加法**：传统正弦-余弦编码是"加法"（位置向量 + 词向量），RoPE 是"旋转"（对 Q/K 向量做旋转）。旋转比加法更干净地保持了向量的范数信息，且自然编码了相对位置。

## 代码示例
```python
import torch
import torch.nn as nn
import math

class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码实现"""
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        # 计算频率 \theta_i = 10000^{-2i/d}
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len, device):
        # 计算位置 m 的 sin/cos 值
        pos = torch.arange(seq_len, device=device).float().unsqueeze(1)
        angles = pos * self.inv_freq.unsqueeze(0)  # (seq_len, dim/2)
        sin, cos = torch.sin(angles), torch.cos(angles)

        # 交替拼接 sin 和 cos
        sin = torch.stack([sin, sin], dim=2).reshape(seq_len, -1)
        cos = torch.stack([cos, cos], dim=2).reshape(seq_len, -1)
        return sin, cos

def apply_rotary_emb(x, sin, cos):
    """对输入张量应用旋转位置编码"""
    # x: (batch, seq_len, n_heads, dim)
    # 将 x 分成两半并交换奇偶维度
    x_rotated = torch.stack([-x[..., 1::2], x[..., ::2]], dim=-1).reshape(x.shape)
    return x * cos + x_rotated * sin

# 使用示例
dim = 64
seq_len = 10
batch_size = 2
n_heads = 8

rope = RotaryPositionalEmbedding(dim)
sin, cos = rope(seq_len, 'cpu')

# 模拟 Q 和 K
Q = torch.randn(batch_size, seq_len, n_heads, dim)
K = torch.randn(batch_size, seq_len, n_heads, dim)

# 应用 RoPE
Q_rotated = apply_rotary_emb(Q, sin, cos)
K_rotated = apply_rotary_emb(K, sin, cos)

# 注意力分数只依赖相对位置
scores = torch.matmul(Q_rotated, K_rotated.transpose(-2, -1))
print(f"RoPE 注意力分数形状: {scores.shape}")  # (batch, seq, n_heads, seq)
```

## 深度学习关联
- **LLaMA 系列的标配**：LLaMA、LLaMA-2、LLaMA-3、Qwen、GLM、Mistral 等模型均使用 RoPE，使其成为目前最主流的位置编码方案。RoPE 对长文本的支持被认为是这些模型成功的关键之一。
- **长度外推的基础**：RoPE 的旋转性质使模型可以外推到训练时未见过的长度。结合 PI (Positional Interpolation)、NTK-aware 等扩展方法，可以在不重新训练的情况下将上下文长度扩展数倍。
- **高效的注意力优化**：RoPE 与 FlashAttention、PagedAttention 等注意力优化完全兼容，因为旋转操作只在 Q/K 输入层面进行，不改变注意力计算的核心逻辑。
