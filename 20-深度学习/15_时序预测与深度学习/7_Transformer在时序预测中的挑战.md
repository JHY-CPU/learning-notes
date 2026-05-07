# 7_Transformer 在时序预测中的挑战

## 1. Transformer 用于时序预测的思路

将时间序列视为"时间 token"序列，直接应用自注意力机制：

```
输入: [x₁, x₂, ..., xₙ] → 嵌入 → 位置编码 → Transformer Encoder → 预测
```

**原始方法：** 将每个时间步作为一个 token，通过自注意力建模全局依赖。

## 2. 核心挑战

### 2.1 二次复杂度问题

自注意力的时间和空间复杂度为 $O(L^2 \cdot d)$，其中 $L$ 为序列长度，$d$ 为特征维度。

| 序列长度 L | 内存占用 (d=512, batch=32) | 实际可行 |
|-----------|--------------------------|---------|
| 96        | ~150 MB                  | 可行    |
| 720       | ~8.5 GB                  | 困难    |
| 2048      | ~68 GB                   | 不可行  |
| 4096+     | >270 GB                  | 不可行  |

```python
import torch
import torch.nn as nn

# 标准自注意力
class StandardAttention(nn.Module):
    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, heads, seq_len, d_k)
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        # scores: (batch, heads, seq_len, seq_len) — 内存瓶颈!
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V)
```

### 2.2 位置编码在时序中的局限

Transformer 本身不感知顺序，需位置编码。但在时间序列中：

**问题 1：** 正弦位置编码适合离散 token（词），时序数据是连续值

**问题 2：** 绝对位置编码无法捕捉相对时间间隔（不等间隔采样）

**问题 3：** 训练长度外推到更长序列时，泛化能力差

```python
# 标准正弦位置编码
def sinusoidal_position_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    position = torch.arange(0, seq_len).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                         -(torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

# 时序中更适合的相对位置编码
class RelativePositionEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.rel_pos_embedding = nn.Embedding(2 * max_len, d_model)

    def forward(self, seq_len):
        positions = torch.arange(seq_len)
        rel_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        rel_positions = rel_positions + seq_len  # 偏移使索引非负
        return self.rel_pos_embedding(rel_positions)
```

### 2.3 时间序列的独特性质

| 时序特性 | NLP 中 | 时序中 | 挑战 |
|---------|--------|--------|------|
| token 含义 | 离散语义 | 连续数值 | 嵌入方式不同 |
| 位置关系 | 线性相邻 | 时间间隔不等 | 位置编码困难 |
| 注意力模式 | 局部+全局 | 主要局部（近期更重要） | 全局注意力浪费 |
| 序列长度 | 几百-几千 | 可能上万 | 复杂度问题 |
| 通道关系 | 无 | 多变量相互影响 | 要不要建模通道？ |

## 3. 简单 Transformer 时序预测实现

```python
class SimpleTransformerForecaster(nn.Module):
    """基础 Transformer 时序预测器（用于理解基本原理）"""
    def __init__(self, input_size, d_model, nhead, num_layers,
                 dim_feedforward, horizon, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = sinusoidal_position_encoding(2048, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        seq_len = x.size(1)
        x = x + self.pos_encoding[:seq_len, :].to(x.device)

        # 因果掩码（防止看到未来信息）
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()

        out = self.encoder(x, mask=causal_mask)
        last = out[:, -1, :]  # 取最后时间步
        return self.fc_out(last)
```

## 4. 注意力模式分析

对时序数据的注意力矩阵分析发现：

```
NLP 注意力:                    时序注意力:
  a  b  c  d  e                t₁ t₂ t₃ t₄ t₅
a █▓▒░░   ← 多样化             t₁ ██████▓░  ← 高度集中于近期
b ░▓█▒░                       t₂ ░█████▓▒
c ░░▒█▓                       t₃ ░░█████▓
d ░░░▓█                       t₄ ░░░█████
e ░░░▒▓█                      t₅ ░░░░█████
  远近都有注意力                  主要关注近期!
```

**发现：** 时序数据的注意力高度集中于最近的时间步，大量注意力计算被浪费。这催生了后续的高效注意力设计。

## 5. 解决方案概览

| 方法 | 核心思想 | 代表模型 |
|------|---------|---------|
| 稀疏注意力 | 只计算部分注意力对 | Informer |
| 自相关 | 用时域自相关替代注意力 | Autoformer |
| 频域注意力 | 在频域计算注意力 | FEDformer |
| 补丁化 | 将序列分段作为 token | PatchTST |
| 倒置 | 变量作为 token | iTransformer |

## 6. 实际使用建议

```python
# 对于中短期预测（≤512步），标准 Transformer 可以工作
model = SimpleTransformerForecaster(
    input_size=7, d_model=128, nhead=8,
    num_layers=3, dim_feedforward=256,
    horizon=24
)

# 对于长期预测或超长序列，使用高效变体
# 推荐: Informer, Autoformer, PatchTST, iTransformer
```

---

**要点总结：**
- 标准 Transformer 的 $O(L^2)$ 复杂度限制了其在长时序上的应用
- 位置编码需要针对时序数据的连续性和不等间隔特性进行改进
- 时序注意力倾向于集中在近期，全局注意力存在冗余
- 需要针对时序特点专门设计高效 Transformer 变体
