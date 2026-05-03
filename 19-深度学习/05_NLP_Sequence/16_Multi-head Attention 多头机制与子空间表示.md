# 16_Multi-head Attention 多头机制与子空间表示

## 核心概念

- **多头注意力 (Multi-head Attention)**：将 Query、Key、Value 分别线性投影到 $h$ 个不同的低维子空间，在每个子空间独立执行注意力计算，然后将结果拼接并投影回原始维度。
- **子空间表示学习**：每个注意力头在不同的投影空间中学习，可能捕捉不同类型的特征——语法关系（如主谓一致）、语义关系（如指代消解）、位置关系等。
- **计算效率**：将 $d_{\text{model}}$ 维的输入投影到 $d_k = d_{\text{model}}/h$ 维的子空间，多头注意力的总计算量等价于单头注意力（$h \cdot d_k^2 = h \cdot (d_{\text{model}}/h)^2 = d_{\text{model}}^2/h$），但实际上通过矩阵乘法实现了并行。
- **头数与模型维度权衡**：常用配置 $h=8$，$d_k=64$（对于 $d_{\text{model}}=512$）。头数太少限制表达能力，太多则每个头的维度太低导致注意力退化。
- **信息整合**：多头的输出拼接后通过 $W_O$ 线性投影整合，实现跨子空间的信息融合。
- **可解释性**：不同注意力头可能关注不同的模式——有些头关注相邻词，有些头关注句法依赖，有些头关注特殊标记位置。这种分工是可解释性研究的重要方向。

## 数学推导

多头注意力计算：
$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O
$$

其中每个头：
$$
\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

各投影矩阵维度：
- $W_Q^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_K^i \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_V^i \in \mathbb{R}^{d_{\text{model}} \times d_v}$
- $W_O \in \mathbb{R}^{h \cdot d_v \times d_{\text{model}}}$

通常 $d_k = d_v = d_{\text{model}} / h$，使得多头总参数与单头一致。

**效率证明**：单头注意力的复杂度为 $O(n^2 d_{\text{model}})$。多头注意力中每个头的复杂度为 $O(n^2 (d_{\text{model}}/h))$，$h$ 个头总计仍为 $O(n^2 d_{\text{model}})$，但每个头在更小的子空间表征中学习。

## 直观理解

- **多头注意力像多领域专家会诊**：对于同一个病人（输入序列），放射科专家（一个注意力头）关注骨骼结构，血液专家关注血常规指标，心脏专家关注心电图……每个专家从各自专业角度分析，最后综合意见得出诊断。
- **子空间投影像戴上不同颜色眼镜**：每个注意力头相当于戴上一副仅透过特定颜色（特征）的眼镜去观察序列。红色眼镜看到语法关系，蓝色眼镜看到语义关系，绿色眼镜看到位置关系。
- **8 头比 1 头好在哪里**：单一注意力头只能将"注意力预算"分配在一组权重上，8 个头可以同时捕捉 8 种不同的关系类型——就像同时有 8 个不同的解释。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 1. 投影并分割头 [batch, seq, d_model] -> [batch, seq, n_heads, d_k]
        Q = self.W_Q(Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(V).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # 3. 拼接所有头并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.d_model)
        output = self.W_O(attn_output)

        return output

# 使用示例
mha = MultiHeadAttention(d_model=512, n_heads=8)
x = torch.randn(4, 20, 512)   # (batch, seq, d_model)
out = mha(x, x, x)             # 自注意力模式
print("多头注意力输出:", out.shape)  # (4, 20, 512)
```

## 深度学习关联

- **Transformer 的核心创新之一**：多头注意力是 Transformer 区别于早期注意力机制的核心设计，使模型能够同时关注不同位置的表示子空间信息。
- **GPT/BERT/LLaMA 的标配**：所有现代 Transformer 模型都使用多头注意力。如 GPT-3 使用 96 个头（d_model=12288, d_k=128），LLaMA 使用 32 个头。
- **头数优化的研究**：后续研究发现并非所有头都同等重要（"注意力头修剪"），有些头可以被移除而不影响性能，催生了模型压缩和架构搜索的相关研究。
