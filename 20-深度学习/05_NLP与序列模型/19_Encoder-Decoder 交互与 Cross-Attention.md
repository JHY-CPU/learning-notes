# 20_Encoder-Decoder 交互与 Cross-Attention

## 核心概念

- **交叉注意力 (Cross-Attention)**：解码器中的注意力层，Query 来自解码器（目标序列），Key 和 Value 来自编码器（源序列）。实现了从目标序列"关注"源序列的机制。
- **信息桥接**：交叉注意力是编码器和解码器之间唯一的交互通道。编码器提取的源序列信息通过这一机制传递给解码器。
- **QKV 的来源差异**：与自注意力（Q=K=V 来自同序列）不同，交叉注意力中 Q 来自解码器，K 和 V 来自编码器。这实现了"跨序列"的信息交互。
- **参数矩阵独立**：交叉注意力的 $W_Q$ 使用解码器输入的投影，$W_K$ 和 $W_V$ 使用编码器输出的投影。交叉注意力不共享自注意力的参数。
- **计算复杂度**：交叉注意力的复杂度为 $O(n_{\text{tgt}} \times n_{\text{src}} \times d)$，其中 $n_{\text{tgt}}$ 是目标序列长度，$n_{\text{src}}$ 是源序列长度。
- **Multi-Head 交叉注意力**：和自注意力一样，交叉注意力也使用多头机制，使解码器能在不同的子空间关注源序列的不同方面。

## 数学推导

交叉注意力的计算：
$$
Q = X_{\text{dec}} W_Q^{(cross)}, \quad K = X_{\text{enc}} W_K^{(cross)}, \quad V = X_{\text{enc}} W_V^{(cross)}
$$

$$
\text{CrossAttn}(X_{\text{dec}}, X_{\text{enc}}) = \text{softmax}\left(\frac{X_{\text{dec}} W_Q (X_{\text{enc}} W_K)^\top}{\sqrt{d_k}}\right) X_{\text{enc}} W_V
$$

注意维度变化：
- $Q \in \mathbb{R}^{n_{\text{tgt}} \times d_k}$（来自解码器）
- $K, V \in \mathbb{R}^{n_{\text{src}} \times d_k}$（来自编码器）
- 注意力矩阵 $\in \mathbb{R}^{n_{\text{tgt}} \times n_{\text{src}}}$（目标关注源）

**无需因果掩码**：交叉注意力不需要因果掩码，因为解码器"看到"完整的编码器输出是合理的——类比翻译时你可以看到完整的源句子。

## 直观理解

- **交叉注意力像同声传译**：译员（解码器）在生成目标语言时，耳朵（交叉注意力）一直听着源语言。每次说目标语言的一个词，都可以选择性地关注源语言的不同部分。
- **架设思维的桥梁**：编码器提取的源序列隐藏状态就像是"源语言记忆笔记"，交叉注意力就是解码器根据需要查阅这些笔记的过程。
- **不对称的 QKV 角色**：解码器状态是"提问者"（Query），编码器状态是"被问者"（Key）和"信息提供者"（Value）。解码器问："现在我该关注源序列的哪个部分？"，编码器回答："这里是相关内容。"

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """交叉注意力层"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        # Q 来自解码器，K/V 来自编码器
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, decoder_states, encoder_states, encoder_mask=None):
        batch_size = decoder_states.size(0)

        Q = self.W_Q(decoder_states).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(encoder_states).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(encoder_states).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # 注意力形状: (batch, n_heads, tgt_seq, src_seq)
        if encoder_mask is not None:
            scores = scores.masked_fill(encoder_mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.n_heads * self.d_k)
        output = self.W_O(attn_output)

        return output, attn_weights

# 交叉注意力示例
cross_attn = CrossAttention(d_model=512, n_heads=8)
decoder_states = torch.randn(4, 15, 512)   # (batch, tgt_seq, d_model)
encoder_states = torch.randn(4, 20, 512)   # (batch, src_seq, d_model)

out, weights = cross_attn(decoder_states, encoder_states)
print("交叉注意力输出:", out.shape)         # (4, 15, 512)
print("注意力权重形状:", weights.shape)     # (4, 8, 15, 20)
# weights[batch, head, tgt_pos, src_pos] = 生成第 tgt_pos 个目标词时
# 对第 src_pos 个源词的关注程度
```

## 深度学习关联

- **Encoder-Decoder 模型的标配**：所有 Encoder-Decoder 架构（原始 Transformer、T5、BART、M2M-100）都在解码器中包含交叉注意力层，这是实现条件生成的核心机制。
- **多模态扩展的基础**：交叉注意力是实现多模态输入的常用方式——编码器可以是图像编码器（ViT）或语音编码器（Whisper），交叉注意力使文本解码器能关注视觉或语音特征。
- **Decoder-only 模型中的替代方案**：GPT 等 Decoder-only 模型不使用独立的交叉注意力，而是将所有任务统一为文本生成。但在检索增强生成 (RAG) 中，检索到的文档通过注意力机制被"交叉关注"。
