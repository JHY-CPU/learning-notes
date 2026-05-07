# 8_Informer：长序列预测

## 1. 概述

Informer (AAAI 2021 Best Paper) 专门解决长序列时序预测 (Long Sequence Time-Series Forecasting, LSTF) 问题，提出三个核心创新：

1. **ProbSparse 自注意力**：$O(L \log L)$ 复杂度
2. **自注意力蒸馏**：层级减半序列长度
3. **生成式解码器**：一次预测整个未来序列

## 2. ProbSparse 自注意力

**核心观察：** 注意力分数分布通常是长尾的——少数 query 有"活跃"注意力，大部分 query 的注意力接近均匀分布（即信息量低）。

**KL 散度度量：** 对每个 query $i$，计算其注意力分布与均匀分布的 KL 散度：

$$M(q_i, K) = \ln \sum_{j=1}^{L_K} e^{\frac{q_i k_j^T}{\sqrt{d}}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^T}{\sqrt{d}}$$

- $M$ 值越大，说明该 query 的注意力越集中（信息量越大）
- $M$ 值越小，说明注意力接近均匀（可用均匀分布近似）

**采样策略：** 选择 Top-u 个"活跃" query（$M$ 值最大），其余用均匀分布近似：

$$\text{Attention}(Q, K, V) \approx \text{softmax}\left(\frac{Q_{\text{sparse}} K^T}{\sqrt{d_k}}\right) V$$

其中 $u = c \cdot \log L_K$，$c$ 为采样因子。

```python
import torch
import torch.nn as nn
import math

class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, factor=5):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor  # 采样因子 c

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _prob_QK(self, Q, K, sample_k):
        """
        计算 Top-u query 的稀疏度量 M
        Q: (batch, heads, L_Q, d_k)
        K: (batch, heads, L_K, d_k)
        """
        B, H, L_Q, d_k = Q.shape
        L_K = K.size(2)

        # 随机采样 K 的子集计算 M 值
        K_sample = K[:, :, torch.randint(L_K, (sample_k,)), :]  # (B,H,sample_k,d_k)

        # 计算 Q 与采样 K 的注意力分数
        Q_K_sample = torch.matmul(Q, K_sample.transpose(-2, -1)) / math.sqrt(d_k)

        # M = ln(sum(exp(score))) - mean(score)
        M = Q_K_sample.logsumexp(dim=-1) - Q_K_sample.mean(dim=-1)
        return M  # (B, H, L_Q)

    def forward(self, Q, K, V):
        B, L_Q, _ = Q.shape
        L_K = K.size(1)

        Q = self.W_q(Q).view(B, L_Q, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, L_K, self.n_heads, self.d_k).transpose(1, 2)

        # 计算采样数
        U = min(self.factor * math.ceil(math.log(L_K)), L_Q)  # Top-u
        u = min(self.factor * math.ceil(math.log(L_K)), L_K)  # sample-k

        # 计算稀疏度量
        M = self._prob_QK(Q, K, u)  # (B, H, L_Q)

        # 选择 Top-U 个 query
        _, M_top_idx = M.topk(U, dim=-1)  # (B, H, U)

        # 对 Top-U query 计算完整注意力
        Q_top = Q.gather(2, M_top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k))
        attn_scores = torch.matmul(Q_top, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        context_top = torch.matmul(attn_weights, V)  # (B, H, U, d_k)

        # 非 Top-U query 的输出用 V 的均值近似
        context_mean = V.mean(dim=2, keepdim=True)  # (B, H, 1, d_k)
        context = context_mean.expand(-1, -1, L_Q, -1).clone()
        context.scatter_(2, M_top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_k),
                         context_top)

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(B, L_Q, -1)
        return self.W_o(context)
```

**复杂度对比：**

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 标准注意力 | $O(L^2 d)$ | $O(L^2)$ |
| ProbSparse | $O(L \log L \cdot d)$ | $O(L \log L)$ |

## 3. 自注意力蒸馏 (Distillation)

将序列长度减半，提取主导特征：

```
Layer 0:  [x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈]   L=8
              ↓ 1D Conv + MaxPool + ELU
Layer 1:  [x₁'   x₃'   x₅'   x₇'      ]   L=4
              ↓ 1D Conv + MaxPool + ELU
Layer 2:  [x₁''         x₅''          ]   L=2
```

```python
class DistillationLayer(nn.Module):
    """Informer 蒸馏层：减半序列长度"""
    def __init__(self, d_model):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3,
                              padding=1, padding_mode='circular')
        self.norm = nn.LayerNorm(d_model)
        self.activation = nn.ELU()

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x.transpose(1, 2)       # (B, d_model, L)
        x = self.conv(x)
        x = self.activation(x)
        # MaxPool 减半序列长度
        x = torch.max_pool1d(x, kernel_size=2, stride=2)
        x = x.transpose(1, 2)       # (B, L/2, d_model)
        return self.norm(x)
```

## 4. 生成式解码器

传统自回归解码需要 $O(H)$ 步推理。Informer 使用**生成式解码**：一次性输入未来序列的占位符，一步输出所有预测值。

```python
class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len,
                 pred_len, d_model=512, n_heads=8, e_layers=3,
                 d_layers=2, d_ff=2048, dropout=0.1, factor=5):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        # 编码器嵌入
        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)

        # 编码器（含蒸馏）
        self.encoder_layers = nn.ModuleList()
        self.distill_layers = nn.ModuleList()
        for i in range(e_layers):
            self.encoder_layers.append(
                nn.TransformerEncoderLayer(d_model, n_heads, d_ff,
                                           dropout, batch_first=True,
                                           activation='gelu')
            )
            if i < e_layers - 1:  # 最后一层不蒸馏
                self.distill_layers.append(DistillationLayer(d_model))

        # 解码器
        decoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, dropout,
            batch_first=True, activation='gelu'
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, d_layers)

        self.projection = nn.Linear(d_model, c_out)

    def forward(self, enc_x, dec_x):
        # enc_x: (B, seq_len, enc_in) 历史观测
        # dec_x: (B, label_len+pred_len, dec_in) 解码器输入

        # 嵌入
        enc_out = self.enc_embedding(enc_x)
        dec_out = self.dec_embedding(dec_x)

        # 编码器 + 蒸馏
        for i, layer in enumerate(self.encoder_layers):
            enc_out = layer(enc_out)
            if i < len(self.distill_layers):
                enc_out = self.distill_layers[i](enc_out)

        # 解码器
        dec_out = self.decoder(dec_out)

        # 只取预测部分的输出
        output = self.projection(dec_out[:, -self.pred_len:, :])
        return output
```

## 5. Informer 架构总览

```
历史序列 ──→ [嵌入] ──→ [Encoder + ProbSparse + Distillation] ──→ 编码特征
                                                                      │
Decoder输入 ──→ [嵌入] ──→ [Decoder + ProbSparse] ←──────────────────┘
                              │
                        [线性投影]
                              │
                         预测序列 y
```

## 6. 实验效果与使用

```python
# Informer 在 ETT 数据集上的典型配置
model = Informer(
    enc_in=7,        # 编码器输入特征数
    dec_in=7,        # 解码器输入特征数
    c_out=1,         # 输出目标维度
    seq_len=96,      # 历史序列长度
    label_len=48,    # 解码器已知部分长度
    pred_len=24,     # 预测长度
    d_model=512,
    n_heads=8,
    e_layers=3,      # 编码器层数（含蒸馏后实际为 L/2^(e-1) 长度）
    d_layers=2,      # 解码器层数
    d_ff=2048,
    factor=5,        # ProbSparse 采样因子
    dropout=0.05
)

# 输入格式
enc_input = torch.randn(32, 96, 7)   # 历史96步
dec_input = torch.randn(32, 72, 7)   # label_len=48 + pred_len=24
output = model(enc_input, dec_input)  # (32, 24, 1)
```

| 预测长度 | 标准 Transformer (MSE) | Informer (MSE) | 提升 |
|---------|----------------------|----------------|------|
| 24      | 0.068                | 0.060          | 12%  |
| 48      | 0.113                | 0.098          | 13%  |
| 168     | 0.182                | 0.152          | 16%  |
| 720     | 0.256                | 0.188          | 27%  |

---

**要点总结：**
- ProbSparse 注意力通过 KL 散度选择 Top-u 个信息量最大的 query，复杂度降至 $O(L \log L)$
- 自注意力蒸馏层级减半序列长度，同时保留关键特征
- 生成式解码器一步输出全部预测值，避免自回归误差累积
- Informer 在长序列预测上显著优于标准 Transformer，且内存消耗大幅降低
