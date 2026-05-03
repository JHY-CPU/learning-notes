# 48_Mistral：滑动窗口注意力机制

## 核心概念

- **Mistral-7B**：由 Mistral AI 于 2023 年 9 月发布，仅 7B 参数但在多项基准上超越了 LLaMA-2-13B 甚至 LLaMA-1-34B。核心创新包括滑动窗口注意力和分组查询注意力。
- **滑动窗口注意力 (Sliding Window Attention, SWA)**：每个 token 只关注相邻 $W$ 个 token（左右各 $W/2$），窗口大小 $W=4096$。将标准注意力的 $O(n^2)$ 复杂度降为 $O(n \cdot W)$。
- **分组查询注意力 (Grouped Query Attention, GQA)**：将多个 Query 头分为一组，每组共享一个 Key 和 Value 头。相比 MHA（每头独立 K/V）减少 KV cache 大小，相比 MQA（所有头共享 K/V）保留更多表达能力。
- **KV cache 显著降低**：由于 GQA 和 SWA 的结合，Mistral-7B 的推理显存需求大幅低于同规模模型，使其在消费级 GPU 上高效运行。
- **Rolling Buffer Cache**：采用滚动缓冲区管理 KV cache，只保留最近 $W$ 个 token 的 KV 值。当序列超过 $W$ 时，最早的 token 被丢弃。这使得固定大小的 cache 可以处理任意长度的序列。
- **Pre-fill and Chunking**：推理时将长序列分成多个块（chunks），逐块填充到滚动缓冲区中。每个块内使用因果掩码，块间保持滑动窗口的连续性。
- **性能与效率的平衡**：尽管使用滑动窗口，但通过多层 Transformer 的堆叠，信息仍然可以在序列中传播（通过多层传递，每个 token 的感受野可达 $L \times W$，$L$ 是层数）。
- **Mistral-Medium 和 Mixtral**：Mistral AI 后续发布了更大的模型和 MoE 版本的 Mixtral 8x7B。

## 数学推导

滑动窗口注意力（窗口大小 $W$，序列长度 $n$）：
$$
\text{SWA}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{swa}}\right)V
$$

滑动窗口掩码：
$$
M_{\text{swa}}(i, j) = \begin{cases}
0 & \text{if } |i - j| \leq W/2 \text{ or } j \leq i \\
-\infty & \text{otherwise}
\end{cases}
$$

**GQA 的参数量**：设 $n_h$ 个 Query 头，$n_{kv}$ 个 Key/Value 头（$n_{kv} \ll n_h$），每组有 $g = n_h / n_{kv}$ 个 Query 头共享一个 KV 头。

KV cache 大小对比（序列 $L$，头维度 $d_h$）：
- MHA：$2 \times L \times n_h \times d_h$
- GQA（$n_{kv} = n_h / g$）：$2 \times L \times n_{kv} \times d_h = \frac{1}{g} \times$ MHA
- MQA（$n_{kv} = 1$）：$2 \times L \times d_h$

Mistral-7B 使用 $n_h = 32$, $n_{kv} = 8$（$g = 4$），KV cache 仅为 MHA 的 25%。

## 直观理解

- **滑动窗口注意力像"有限视野"**：你走在街上，视野范围有限（窗口大小 $W$），只能看清周围一定距离的人和物。但你边走边看（多层 Transformer），虽然每步的视野有限，但通过积累可以了解整个街区的全貌。
- **GQA 像"小组分享笔记"**：MHA 是每个人都做自己的笔记（每头独立 K/V），MQA 是全班共用一份笔记（所有头共享 K/V）。GQA 折中——小组内共享笔记。4 个同学一组，每组共享一份笔记，减少了笔记总量但保留了小组的个性化。
- **Rolling Buffer 像"传送带"**：KV cache 就像一个传送带，最新的 token 放在右侧，超过窗口宽度的旧 token 从左侧掉下去。传送带长度固定，但可以处理无限长的输入序列。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力（简化实现）"""
    def __init__(self, d_model, n_heads, n_kv_heads, window_size=4096):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.window_size = window_size

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model // (n_heads // n_kv_heads), bias=False)
        self.W_V = nn.Linear(d_model, d_model // (n_heads // n_kv_heads), bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        batch, seq, _ = x.shape
        Q = self.W_Q(x).view(batch, seq, self.n_heads, self.d_k)
        # GQA: 只有 n_kv_heads 个 K, V
        K = self.W_K(x).view(batch, seq, self.n_kv_heads, self.d_k)
        V = self.W_V(x).view(batch, seq, self.n_kv_heads, self.d_k)

        # GQA: 将 K, V 扩展到 n_heads (每组复制)
        K = K[:, :, :, None, :].expand(-1, -1, -1, self.n_groups, -1)
        K = K.reshape(batch, seq, self.n_heads, self.d_k)
        V = V[:, :, :, None, :].expand(-1, -1, -1, self.n_groups, -1)
        V = V.reshape(batch, seq, self.n_heads, self.d_k)

        Q = Q.transpose(1, 2)  # (batch, n_heads, seq, d_k)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # SWA 掩码
        if self.window_size:
            swa_mask = torch.ones(seq, seq, device=x.device)
            for i in range(seq):
                start = max(0, i - self.window_size // 2)
                end = min(seq, i + self.window_size // 2 + 1)
                swa_mask[i, :start] = 0
                swa_mask[i, end:] = 0
            swa_mask = swa_mask.unsqueeze(0).unsqueeze(0)
            scores = scores.masked_fill(swa_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return self.W_O(out)

# Mistral-7B 架构参数
print("Mistral-7B 架构参数:")
print(f"  hidden_dim: 4096")
print(f"  n_layers: 32")
print(f"  n_heads: 32, n_kv_heads: 8 (GQA, groups=4)")
print(f"  window_size: 4096 (SWA)")
print(f"  KV cache 节省: {(1 - 8/32) * 100:.0f}% vs MHA")

# 模拟滑动窗口注意力的感受野
n_layers = 32
window_size = 4096
num_tokens = 32 * 128  # 一个长序列
max_distance_reached = window_size * n_layers
print(f"\n通过 {n_layers} 层堆叠，每个 token 的理论感受野: {max_distance_reached:,}")
print("实际效果受限于信息传播效率，通常为 O(W * sqrt(L))")
```

## 深度学习关联

- **高效推理的工程典范**：Mistral-7B 展示了如何通过架构设计（SWA + GQA + Rolling Buffer）实现在消费级硬件上的高效推理，是"小模型 + 好架构 > 大模型"思想的成功案例。
- **滑动窗口的局限与改进**：滑动窗口对于需要长距离精确依赖的任务（如长文档的指代消解、书籍级推理）可能不足。后续的 LongLoRA、YaRN 等扩展 SWA 的方案尝试解决这一问题。
- **Mixtral 8x7B (MoE)**：Mistral AI 后续发布的 Mixtral 使用 MoE 架构，在保持推理高效的同时大幅增加了模型容量，在 12.9B 激活参数下达到了 70B 级别模型的性能。
