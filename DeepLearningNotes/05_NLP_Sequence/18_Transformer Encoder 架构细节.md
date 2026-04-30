# 18_Transformer Encoder 架构细节

## 核心概念
- **Transformer 编码器**：由 Vaswani et al. (2017) 提出，由 N 个相同层堆叠而成，每层包含两个子层：多头自注意力 (Multi-Head Self-Attention) 和前馈神经网络 (FFN)。
- **残差连接 (Residual Connection)**：每个子层的输出都加上其输入：$\text{LayerNorm}(x + \text{Sublayer}(x))$。残差连接使梯度可以直接流过深层的编码器堆叠。
- **层归一化 (Layer Normalization)**：对每个样本的特征维度做归一化（均值为 0、方差为 1），代替批归一化。LayerNorm 在序列模型中更稳定，因为序列长度可变。
- **逐位置前馈网络 (Position-wise FFN)**：由两个线性变换组成，中间使用 ReLU 激活：$\text{FFN}(x) = W_2 \cdot \max(0, W_1 x + b_1) + b_2$。内层维度通常扩展 4 倍（$d_{\text{ff}} = 4 \times d_{\text{model}}$）。
- **编码器的输入输出**：输入为词嵌入 + 位置编码，输出为同维度的上下文表示。编码器的输出包含每个位置整合了全序列信息的向量。
- **Post-Norm vs Pre-Norm**：原始 Transformer 使用 Post-Norm（残差后归一化），但现代实现（如 GPT、LLaMA）更常用 Pre-Norm（残差前归一化），使训练更稳定。
- **Dropout**：在每个子层输出、嵌入、注意力权重上应用 Dropout 防止过拟合。

## 数学推导
编码器第 $l$ 层的计算：

$$
Q = K = V = X^{l-1}
$$

$$
X' = \text{LayerNorm}(X^{l-1} + \text{MultiHeadAttn}(X^{l-1}))
$$

$$
X^l = \text{LayerNorm}(X' + \text{FFN}(X'))
$$

其中 $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$。

对于 Pre-Norm 变体：
$$
X' = X^{l-1} + \text{MultiHeadAttn}(\text{LayerNorm}(X^{l-1}))
$$

$$
X^l = X' + \text{FFN}(\text{LayerNorm}(X'))
$$

编码器的信息流：
- 自注意力层：每个位置聚合所有位置的信息，$O(n^2 d)$
- FFN 层：每个位置独立进行非线性变换，$O(n d^2)$
- 总复杂度：$O(n^2 d + n d^2)$

## 直观理解
- **编码器像文本分析工作室**：自注意力层是"信息交流大厅"——每个词与所有词交换信息（语法角色、语义关系等）。FFN 层是"私人办公室"——每个词在获得全局信息后，独自进行深度思考（非线性变换）。
- **残差连接像高速公路**：在拥挤的信息处理流程（多层网络）中，残差连接就像一条直达高速公路，让原始信息轻松绕过复杂处理环节直达深层。
- **LayerNorm 像数据标准化**：确保每层的信息分布都"正常化"（均值为 0、方差为 1），避免数据分布偏移导致训练不稳定。
- **逐位置 FFN 的含义**：虽然"逐位置"听起来独立，但由于前一层是自注意力（它已经聚合了上下文信息），FFN 接收的每个位置的向量已经是"具有全局感知的局部表示"。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器单层实现"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # 自注意力 + 残差 + LayerNorm
        attn_out, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_out))

        # FFN + 残差 + LayerNorm
        ff_out = self.linear2(F.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout(ff_out))

        return src

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, d_ff=2048,
                 num_layers=6, max_len=512, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._create_pos_encoding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def _create_pos_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.pos_encoding[:, :seq_len, :].to(x.device)
        x = self.dropout(x)

        # Transformer 期望输入形状 (seq_len, batch, d_model)
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x)

        # 转回 (batch, seq_len, d_model)
        x = x.transpose(0, 1)
        return x

# 使用示例
encoder = TransformerEncoder(vocab_size=10000, d_model=512, num_layers=6)
x = torch.randint(0, 10000, (4, 20))   # (batch, seq_len)
output = encoder(x)
print("编码器输出:", output.shape)      # (4, 20, 512)
```

## 深度学习关联
- **BERT 和 RoBERTa 的基础**：BERT 使用多层 Transformer Encoder 作为骨干网络，通过 MLM 预训练学习深度双向表示。现代 Encoder-only 模型都遵循这一架构。
- **Encoder-Decoder 模型中的编码器**：T5、BART 等模型中的编码器与标准 Transformer Encoder 一致，负责将输入序列编码为上下文表示供解码器使用。
- **架构演进的起点**：原始 Transformer 6 层编码器的配置已被超越——现代模型中编码器深度大幅增加（BERT-large 24 层），且 Pre-Norm、GELU 激活等改进已成为新标准。
