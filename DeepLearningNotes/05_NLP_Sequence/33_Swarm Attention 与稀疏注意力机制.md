# 33_Swarm Attention 与稀疏注意力机制

## 核心概念
- **稀疏注意力 (Sparse Attention)**：为了解决标准自注意力 $O(n^2)$ 复杂度的问题，只计算部分选定的位置对的注意力，而非全部位置对。在长序列场景中至关重要。
- **滑动窗口注意力 (Sliding Window Attention)**：每个 token 只关注其相邻 $w$ 个邻居（左右各 $w/2$）。复杂度从 $O(n^2)$ 降至 $O(n \cdot w)$，其中 $w \ll n$。
- **全局注意力 (Global Attention)**：选择少量特殊 token（如 [CLS] 标记、重要词汇）作为"全局节点"，它们关注所有位置，且所有位置关注它们。确保长距离信息流通。
- **Swarm Attention**：一种最近的稀疏注意力变体，利用"群体智能"的思想。通过分组和选举机制，让 token 在组内共享注意力模式，减少冗余计算。
- **BigBird 的三种注意力组合**：滑动窗口注意力 + 全局注意力 + 随机注意力（每个 token 随机关注 $r$ 个随机位置）。理论上保持图连通性，保证信息能够在任意两个位置之间传播。
- **Longformer 的注意力模式**：滑动窗口 + 全局注意力，更简单的组合。使用"扩张窗口"进一步扩展感受野。
- **稀疏注意力的图论视角**：将注意力视为有向图，稀疏注意力相当于在图上选择关键边。好的稀疏模式需要保证图的连通性和信息流动效率。

## 数学推导
标准自注意力复杂度：$O(n^2 d)$

**滑动窗口注意力**：
$$
\text{Attn}(x_i) = \sum_{j \in [i-w/2, i+w/2]} \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j
$$

复杂度：$O(n \cdot w \cdot d)$，$w$ 是窗口大小。

**BigBird 的稀疏注意力复杂度**：
$$
O(n \cdot w) + O(n \cdot g) + O(n \cdot r) = O(n)
$$

其中 $w$ 是窗口大小，$g$ 是全局 token 数（通常 $g \ll n$），$r$ 是随机 token 数。

**图论保证**：图 $G$ 的直径（任意两节点间最短路径的最大值）为 $O(\log n)$，保证信息可高效流通。

## 直观理解
- **稀疏注意力像"选择性社交"**：标准注意力就像在一场派对中和每个人逐一握手（$n^2$ 次）。滑动窗口注意力只和身边几个人聊天（$w$ 次），全局注意力让派对主持人（全局 token）和每个人交流。你不必认识所有人，但通过信息传递，消息仍能传遍全场。
- **Swarm Attention 像"代表投票"**：一群人分成若干小组，每个小组选举一个代表，只有代表之间进行交流，代表再把信息带回自己小组。这比所有人都互相交流高效得多。
- **随机注意力的奇妙作用**：为什么需要随机连接？就像社交网络中，你不仅认识邻居和名人（全局节点），还偶然认识几个远方的朋友（随机连接），这些"弱连接"有时是传播新信息的关键渠道。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力实现"""
    def __init__(self, d_model, n_heads, window_size=4):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.window_size = window_size
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        Q = self.W_Q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 构建滑动窗口掩码
        mask = torch.ones(seq_len, seq_len, device=x.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, :start] = 0
            mask[i, end:] = 0
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, seq)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_O(out)

# 对比标准注意力和滑动窗口注意力的复杂度
window_size = 128
seq_len = 4096
standard_cost = seq_len ** 2
sw_cost = seq_len * window_size
print(f"标准注意力计算量: {standard_cost:,}")
print(f"滑动窗口注意力计算量: {sw_cost:,}")
print(f"节省比例: {(1 - sw_cost / standard_cost) * 100:.2f}%")

# 滑动窗口示例
sw_attn = SlidingWindowAttention(d_model=256, n_heads=4, window_size=10)
x = torch.randn(2, 50, 256)
out = sw_attn(x)
print(f"\n滑动窗口注意力输出: {out.shape}")
```

## 深度学习关联
- **长文本处理的关键技术**：稀疏注意力使 Transformer 能够处理文档级（数千上万 token）的文本，Longformer、BigBird 分别在 PubMed 的法律文档和长文档理解中取得突破。
- **Mistral 的滑动窗口成功**：Mistral-7B 使用滑动窗口注意力（窗口大小 4096），在保持性能的同时大幅降低了计算量，证明了稀疏注意力在大语言模型中的实用性。
- **与 FlashAttention 的互补**：稀疏注意力和 FlashAttention 解决的是不同维度的问题——稀疏注意力减少 FLOPs，FlashAttention 优化 IO 效率。两者可以结合使用，实现更高效率的长文本处理。
