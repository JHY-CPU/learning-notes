# 32_Alibi (ALiBi)：注意力线性偏置

## 核心概念

- **ALiBi (Attention with Linear Biases)**：由 Press et al. (2021) 提出，通过向注意力分数添加一个线性偏置来编码位置信息，而不是在词嵌入中加入位置编码。
- **核心思想**：注意力分数 $q_i \cdot k_j$ 加上一个与距离 $|i-j|$ 成比例的负偏置 $m \cdot |i-j|$，使得距离越远的词对获得的注意力分数越低。
- **零额外参数**：ALiBi 不引入任何额外的可学习参数，仅需要一个预定义的斜率 $m$。
- **多头差异化斜率**：不同注意力头使用不同的斜率 $m_h$，从几何序列中取值。头越多，最大的斜率越小，最小的斜率也小。
- **长度外推优势**：ALiBi 在长度外推上表现优异——在短序列（如 512）上训练的模型可以直接推广到长序列（如 1024、2048），而不需要位置插值。
- **简单高效**：相比 RoPE，ALiBi 的实现在计算上更简单——只需要在注意力分数矩阵上添加一个静态偏置矩阵。
- **理论解释**：ALiBi 可以看作是在位置编码空间中施加一个"局部性先验"——相邻位置的 token 对当前词更重要，这种先验通过偏置自然实现。

## 数学推导

ALiBi 的注意力计算：
$$
\text{ALiBi}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + B\right)V
$$

其中偏置矩阵 $B$ 定义为：
$$
B_{i,j} = -m_h \cdot |i - j|
$$

$m_h$ 是第 $h$ 个注意力头的斜率：
$$
m_h = 2^{-8h / H}, \quad h = 1, 2, \ldots, H
$$

例如，$H=8$ 时斜率分别为 $m = [1/2^1, 1/2^2, \ldots, 1/2^8]$。

对于距离 $d$（$d_{max}=512$），偏置 $B_{i,j} = -m_h \cdot d$ 是一个单调递减的线性函数。距离越远，偏置越负，注意力分数越低。

**与外推的关系**：当序列长度扩展到 $L' > L$ 时，$B_{i,j}$ 对于 $|i-j| > L$ 的部分只是简单的线性延伸，不需要任何插值。模型在短距离上的行为可以平滑地推广到长距离。

## 直观理解

- **ALiBi 像"近距离的优先关注"**：你与人交谈时，通常更关注刚说的内容（近距离），而不是很久以前说的（远距离）。ALiBi 给每个词对加上一个"距离惩罚"——越远惩罚越大。
- **没有位置编码的 Transformer**：ALiBi 证明了一个令人惊讶的事实——你不需要将位置信息"嵌入"到词向量中，只需在注意力分数上施加位置偏置就足够了。这就像在社交场合，不需要在每个人的名片上注明"第几个来的"，只需通过座位距离就可以知道交流的亲疏。
- **外推优势的来源**：对于短序列来说，$B_{i,j}$ 的最大值对应训练时见过的最大距离。当序列变长时，$B_{i,j}$ 对更大距离的惩罚只是已有模式的线性延伸——模型已经学过了"距离越远，偏置越负"的规律，可以自然地泛化。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ALiBiAttention(nn.Module):
    """实现 ALiBi 偏置的注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

        # 计算多头斜率
        self.register_buffer('slopes', self._get_slopes(n_heads))

    def _get_slopes(self, n_heads):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (2 ** (-i)) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            # 非 2 的幂的情况
            n = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(n)
            extra = get_slopes_power_of_2(2 * n)
            return torch.tensor(slopes + extra[:n_heads - n])

    def _build_alibi_bias(self, seq_len, device):
        # 构建偏置矩阵 B
        pos = torch.arange(seq_len, device=device).float()
        dist = pos.unsqueeze(0) - pos.unsqueeze(1)  # (seq, seq)
        dist = dist.abs()
        # (n_heads, seq, seq)
        bias = -self.slopes.unsqueeze(1).unsqueeze(1) * dist.unsqueeze(0)
        return bias

    def forward(self, Q, K, V):
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        Q = self.W_Q(Q).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 添加 ALiBi 偏置
        bias = self._build_alibi_bias(seq_len, Q.device)
        scores = scores + bias.unsqueeze(0)  # (batch, n_heads, seq, seq)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_O(out)

# 演示 ALiBi 偏置
alibi = ALiBiAttention(d_model=256, n_heads=4)
seq_len = 10
x = torch.randn(2, seq_len, 256)
out = alibi(x, x, x)
print("ALiBi 注意力输出:", out.shape)

# 展示偏置矩阵
bias = alibi._build_alibi_bias(5, 'cpu')
print(f"\n第一头的偏置矩阵 (seq=5):")
print(bias[0])
```

## 深度学习关联

- **长度外推的实用方案**：ALiBi 是少数在工业界得到广泛应用的"零开销"位置编码方案之一，在 BLOOM 等大模型中被采用，证明了简单方法的有效性。
- **位置编码思路的分水岭**：ALiBi 代表了一种"位置编码不需要嵌入到词向量中"的激进思想，与 RoPE 等方案形成了对比。后续的 Sandwich 等工作探索了两者的融合。
- **推理效率的考量**：ALiBi 的静态偏置矩阵可以预先计算并缓存，在推理时不需要额外计算，对于 KV cache 高效部署特别友好。
