# 19_Transformer Decoder 与 Causal Masking

## 核心概念
- **Transformer 解码器**：由 N 个相同层堆叠，每层包含三个子层：掩码多头自注意力 (Masked Multi-Head Self-Attention)、交叉注意力 (Cross-Attention)、前馈网络 (FFN)。
- **因果掩码 (Causal Masking / Look-Ahead Mask)**：确保解码器在生成第 $t$ 个位置时只能看到第 $1$ 到 $t$ 个位置，不能看到未来的 token。通过将上三角矩阵设置为 $-\infty$ 实现。
- **自回归生成 (Autoregressive Generation)**：解码器逐步生成 token——用已生成的 token 预测下一个 token。这是因果掩码的直接应用。
- **交叉注意力 (Cross-Attention)**：第二个子层——Query 来自解码器，Key 和 Value 来自编码器。使解码器在生成每个词时能"关注"输入序列的对应部分。
- **与编码器的结构差异**：编码器有 2 个子层（自注意力 + FFN），解码器有 3 个子层（掩码自注意力 + 交叉注意力 + FFN）。解码器多了一个交叉注意力子层。
- **训练 vs 推理**：训练时使用 Teacher Forcing（直接输入完整目标序列，掩码保证因果性），推理时逐个生成 token。

## 数学推导
因果掩码的数学表达：
$$
\text{MaskedAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中掩码矩阵 $M \in \mathbb{R}^{n \times n}$ 定义为：
$$
M_{ij} = \begin{cases}
0 & \text{if } j \leq i \\
-\infty & \text{if } j > i
\end{cases}
$$

softmax 之后，$j > i$ 的位置权重变为 0，实现单向信息流。

解码器第 $l$ 层的完整计算：

1. **掩码自注意力**：
$$
Q_1 = K_1 = V_1 = X^{l-1}
$$
$$
X_1 = \text{LayerNorm}(X^{l-1} + \text{MaskedMultiHeadAttn}(Q_1, K_1, V_1))
$$

2. **交叉注意力**：
$$
Q_2 = X_1,\quad K_2 = V_2 = H_{\text{enc}}
$$
$$
X_2 = \text{LayerNorm}(X_1 + \text{MultiHeadAttn}(Q_2, K_2, V_2))
$$

3. **前馈网络**：
$$
X^l = \text{LayerNorm}(X_2 + \text{FFN}(X_2))
$$

## 直观理解
- **因果掩码像写文章**：你写文章时只能看到已写的内容，不能提前看"明天的自己会写什么"。每一个新词只能基于已经写出的内容来生成。
- **训练时的"作弊" vs 推理时的诚实**：训练时虽然完整的目标序列已经给出（Teacher Forcing），但掩码强制模型只在已知前缀上做预测——就像闭卷考试时不让你偷看后面的答案。推理时则逐词生成，完全诚实。
- **交叉注意力像查资料**：写英文文章时（解码），你随时查阅中文原文（编码器的输出）。编码器的每个隐藏状态就像原文的"笔记"，交叉注意力决定了此刻该参考笔记的哪个部分。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_causal_mask(seq_len):
    """创建因果掩码矩阵"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask.masked_fill(mask == 1, float('-inf'))

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        # 1. 掩码自注意力
        attn_out, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        # 2. 交叉注意力
        attn_out, _ = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(attn_out))

        # 3. FFN
        ff_out = self.linear2(F.relu(self.linear1(tgt)))
        tgt = self.norm3(tgt + self.dropout(ff_out))

        return tgt

# 模拟推理过程：逐步生成
seq_len = 5
d_model = 128
causal_mask = create_causal_mask(seq_len)
print("因果掩码矩阵:")
print(causal_mask)

# 解码器前向传播（训练模式——一次性输入）
decoder_layer = TransformerDecoderLayer(d_model=128, n_heads=4, d_ff=512)
tgt = torch.randn(seq_len, 4, 128)  # 完整目标序列
memory = torch.randn(10, 4, 128)     # 编码器输出
out = decoder_layer(tgt, memory, tgt_mask=causal_mask)
print("解码器输出:", out.shape)       # (seq_len, batch, d_model)
```

## 深度学习关联
- **GPT 系列的核心架构**：GPT 系列模型（GPT-2、GPT-3、GPT-4）是纯粹的解码器架构，使用因果掩码实现自回归语言建模。不需要编码器，因为始终是文本生成任务。
- **Encoder-Decoder vs Decoder-only**：T5、BART 等 Encoder-Decoder 模型适合需要理解输入的任务（翻译、摘要），而 GPT 等 Decoder-only 模型通过"下一个 token 预测"统一了所有 NLP 任务。
- **Flamingo 等视觉-语言模型**：LLaMA、Flamingo 等模型在解码器中加入交叉注意力或适配器层，使语言模型能够"看"图像或视频输入——交叉注意力是实现多模态融合的关键机制。
