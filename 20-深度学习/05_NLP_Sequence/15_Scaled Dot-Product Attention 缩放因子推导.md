# 15_Scaled Dot-Product Attention 缩放因子推导

## 核心概念

- **Scaled Dot-Product Attention**：Transformer 使用的注意力机制，通过 $\frac{QK^\top}{\sqrt{d_k}}$ 计算注意力分数，其中除以 $\sqrt{d_k}$ 是关键缩放操作。
- **缩放因子的必要性**：当维度 $d_k$ 很大时，点积的数值可能变得极大。这会使 softmax 后的梯度极小（进入饱和区），阻碍训练。
- **点积的方差分析**：假设 $q$ 和 $k$ 是均值为 0、方差为 1 的独立随机变量，则点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 的均值为 0、方差为 $d_k$。
- **缩放后分布稳定**：除以 $\sqrt{d_k}$ 后，点积的方差恢复到 1，使 softmax 的输入处于合适的梯度敏感区间。
- **与未缩放注意力的对比**：不加缩放时，大的 $d_k$ 导致 softmax 输出极度接近 one-hot，注意力过于尖锐，梯度消失。
- **数值稳定性的影响**：点积过大不仅影响梯度，还可能导致数值溢出（softmax 中 $\exp$ 的大输入）。

## 数学推导

设 $q, k \in \mathbb{R}^{d_k}$ 的分量独立同分布，$E[q_i] = E[k_i] = 0$，$\text{Var}(q_i) = \text{Var}(k_i) = 1$。

计算点积的期望和方差：
$$
E[q_i k_i] = E[q_i]E[k_i] = 0
$$

$$
E[q \cdot k] = \sum_{i=1}^{d_k} E[q_i k_i] = 0
$$

$$
\text{Var}(q_i k_i) = E[(q_i k_i)^2] - E[q_i k_i]^2 = E[q_i^2]E[k_i^2] = 1
$$

$$
\text{Var}(q \cdot k) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k
$$

因此点积的标准差为 $\sqrt{d_k}$。当 $d_k$ 较大时，点积的数值范围很广。

除以 $\sqrt{d_k}$ 后：
$$
\text{Var}\left(\frac{q \cdot k}{\sqrt{d_k}}\right) = \frac{d_k}{d_k} = 1
$$

**对 softmax 的影响**：当 $x_i$ 值很大时，$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum e^{x_j}}$ 中最大值项主导，梯度接近 0（因为 $\frac{\partial \text{softmax}}{\partial x_i} \approx 0$ 对非最大项）。缩放后 $x_i$ 在合理范围内，梯度得到保留。

## 直观理解

- **缩放因子像音量调节**：想象一群人在房间里聊天（计算注意力），每个 Q 和 K 对话一次。当 $d_k$ 很大时，就像每人拿扩音器说话——音量过大（点积分值大），信号失真（softmax 饱和）。$\sqrt{d_k}$ 就是音量旋钮，调回适宜水平。
- **方差控制**：如果不缩放，假设 $d_k=512$，点积的方差为 512，意味着典型值在 $\pm 22.6$ 附近（$\sqrt{d_k}$），softmax 后几乎只有最大值有非零概率。缩放后方差为 1，softmax 分布更均匀、梯度更大。
- **从数学到直觉**：$d_k$ 是"注意力"的维度数。维度越高，随机向量点积的波动越大，就像高维空间中几乎所有向量都近似正交——少数非正交的向量得分极高。缩放因子抑制了这种维度导致的统计偏差。

## 代码示例

```python
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V):
    """手动实现缩放点积注意力"""
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (..., seq_q, seq_k)
    scaled_scores = scores / math.sqrt(d_k)         # 缩放
    attn_weights = F.softmax(scaled_scores, dim=-1)
    return torch.matmul(attn_weights, V)

# 演示缩放的作用
d_k_small = 8
d_k_large = 512

Q = torch.randn(1, 5, d_k_large)
K = torch.randn(1, 5, d_k_large)

# 不加缩放
scores_no_scale = torch.matmul(Q, K.transpose(-2, -1))
# 加缩放
scores_scaled = scores_no_scale / math.sqrt(d_k_large)

print(f"d_k={d_k_large} 时:")
print(f"未缩放分数的范围: [{scores_no_scale.min().item():.2f}, {scores_no_scale.max().item():.2f}]")
print(f"缩放后分数的范围: [{scores_scaled.min().item():.2f}, {scores_scaled.max().item():.2f}]")

# 检查 softmax 后梯度的"锐利程度"
softmax_no_scale = F.softmax(scores_no_scale, dim=-1)
softmax_scaled = F.softmax(scores_scaled, dim=-1)

print(f"\n未缩放 softmax 最大权重: {softmax_no_scale.max().item():.4f}")
print(f"缩放后 softmax 最大权重: {softmax_scaled.max().item():.4f}")
# 未缩放的 softmax 输出接近 one-hot，缩放后分布更均匀
```

## 深度学习关联

- **Transformer 的核心操作**：Scaled Dot-Product Attention 是 Transformer 中最基础的注意力单元，所有注意力变体（Multi-head、Cross-attention、Self-attention）都基于此。
- **数值稳定性与训练收敛**：缩放因子的引入使得 Transformer 可以在更大的学习率范围稳定训练，是训练深层 Transformer 的关键工程细节之一。
- **高效计算的工程优化**：FlashAttention 等高效注意力算法仍然使用缩放点积，但通过分块计算和重计算优化了 $O(n^2)$ 的显存占用。
