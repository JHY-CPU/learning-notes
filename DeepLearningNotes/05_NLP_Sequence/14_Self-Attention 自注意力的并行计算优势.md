# 14_Self-Attention 自注意力的并行计算优势

## 核心概念
- **自注意力 (Self-Attention)**：在序列内部计算注意力，每个位置关注序列中的所有其他位置。Query、Key、Value 都来自同一个输入序列，因此得名"自"注意力。
- **并行计算**：自注意力在每个时间步的计算不依赖其他时间步（不像 RNN 需要 $h_{t-1}$ 才能算 $h_t$），因此可以一次性计算所有位置的注意力输出。
- **全局感受野**：自注意力允许每个位置直接关注任意距离的其他位置（步长为 O(1)），而 RNN 需要 O(距离) 步才能传递信息。
- **复杂度分析**：自注意力的计算复杂度是 $O(n^2 d)$，其中 $n$ 是序列长度，$d$ 是特征维度。对于长序列，计算和内存开销大（Transformer 的主要局限之一）。
- **多头机制**：自注意力通常与多头机制结合，让模型在不同表示子空间关注不同的交互模式（语法、语义、位置等）。
- **对位置不敏感**：自注意力是"排列等变"的——打乱输入顺序会得到打乱后的输出，但输入-输出对应关系不变。因此必须单独加入位置编码。
- **矩阵化计算**：自注意力通过矩阵乘法一次完成所有计算，充分利用 GPU 并行能力。

## 数学推导
自注意力的矩阵化计算：
$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中 $X \in \mathbb{R}^{n \times d_{\text{model}}}$ 是输入序列，$W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k}$。

**并行计算的关键**：$QK^\top$ 一次性计算了所有 $n^2$ 个注意力分数对，无需任何循环。$n$ 个 Query 同时与 $n$ 个 Key 做点积：

$$
(QK^\top)_{ij} = q_i \cdot k_j
$$

**与 RNN 的复杂度对比**：
- RNN（单步）：$O(d^2)$，但必须串行执行 $n$ 步 $\Rightarrow$ 总时间 $O(n d^2)$
- 自注意力：$O(n^2 d)$ 全部并行执行 $\Rightarrow$ 总时间 $O(1)$（完全并行）

## 直观理解
- **自注意力像全连接会议**：会议室里每个人（每个 token）同时发言并听所有人说话，然后决定自己"吸收"谁的发言最多（注意力权重）。不需要一个个轮流发言（RNN 的串行模式）。
- **并行 vs 串行**：RNN 读句子像逐个单词朗读——必须按顺序从左到右。自注意力像一目十行地扫读——同时看到所有词，直接理解全局语义。
- **全局连接的解释力**：在"那只猫不关在笼子里，它跑出来了"这句话中，"它"指代"猫"。自注意力可以让"它"直接连接到"猫"（无论距离多远），一步到位理解指代关系。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, embed_size)
        K = self.key(x)      # (batch, seq, head_size)
        Q = self.query(x)    # (batch, seq, head_size)
        V = self.value(x)    # (batch, seq, head_size)

        # 计算注意力分数 (完全并行)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq, seq)
        attn_scores = attn_scores / (K.size(-1) ** 0.5)     # 缩放

        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_weights, V)                 # (batch, seq, head_size)
        return out, attn_weights

# 对比 RNN 串行和自注意力并行
x = torch.randn(4, 10, 128)   # batch=4, seq=10, embed=128

self_attn = SelfAttention(128, 64)
out, weights = self_attn(x)
print("自注意力输出:", out.shape)      # (4, 10, 64)
print("注意力矩阵:", weights.shape)    # (4, 10, 10)
# 注意力矩阵权重显示每个词（行）关注其他词（列）的程度
```

## 深度学习关联
- **Transformer 的核心创新**：自注意力是 Transformer 取代 RNN 的关键原因。并行计算使训练时间从 $O(n d^2)$ 降至可并行化的 $O(1)$ 步，配合 GPU 可处理海量数据。
- **BERT/GPT 的基石**：BERT 使用双向自注意力（可看到所有位置），GPT 使用因果自注意力（只能看到左侧）。两种变体都建立在自注意力机制之上。
- **长序列挑战**：自注意力的 $O(n^2)$ 复杂度是其主要瓶颈。由此催生了 Longformer（滑动窗口 + 全局注意力）、FlashAttention（IO 感知）、稀疏注意力等改进方案。
