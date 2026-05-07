# 12_iTransformer：倒置 Transformer

## 1. 核心创新

iTransformer (ICLR 2024) 提出一个颠覆性观点：**将每个变量的整个时间序列作为一个 token**，而非将每个时间步作为一个 token。

```
传统 Transformer (Token = 时间步):
  输入: [x₁, x₂, ..., x_L] 每个时间步是一个 token
  关注: 时间步之间的依赖

iTransformer (Token = 变量):
  输入: [var₁, var₂, ..., var_D] 每个变量是一个 token
  关注: 变量之间的相关性

  变量1 token: [x₁, x₂, x₃, ..., x_L] → 嵌入 → Transformer
  变量2 token: [x₁, x₂, x₃, ..., x_L] → 嵌入 → Transformer
  ...
  变量D token: [x₁, x₂, x₃, ..., x_L] → 嵌入 → Transformer
```

## 2. 动机分析

### 传统方法的问题

| 方法 | Token 定义 | 建模内容 | 局限 |
|------|-----------|---------|------|
| 标准 Transformer | 时间步 | 时间步间关系 | $O(L^2)$ 复杂度 |
| PatchTST | 补丁 | 时间模式 | 通道独立，忽略变量关系 |
| Autoformer | 时间步 | 时域自相关 | 复杂度仍为 $O(L)$ token |

### iTransformer 的优势

- Token 数 = 变量数 $D$，通常 $D \ll L$，注意力矩阵 $D \times D$ 远小于 $L \times L$
- 自然建模多变量间的相关性
- 每个 token 包含完整的时序信息，嵌入层可以提取丰富的时序特征

## 3. 架构设计

```python
import torch
import torch.nn as nn

class VariableTokenEmbedding(nn.Module):
    """将每个变量的完整时间序列嵌入为一个 token"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        # 使用多层 MLP 或 1D CNN 提取时序特征
        self.embedding = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, L, D) -> 转置为 (B, D, L) 使每个变量成为一个 token
        x = x.transpose(1, 2)  # (B, D, L)
        return self.embedding(x)  # (B, D, d_model)


class VariableTokenEmbeddingCNN(nn.Module):
    """使用 1D CNN 提取变量的时序特征"""
    def __init__(self, seq_len, d_model, dropout=0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 全局平均池化
        )
        self.proj = nn.Linear(64, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, D, L)
        B, D, L = x.shape
        x = x.reshape(B * D, 1, L)  # (B*D, 1, L)
        x = self.conv(x).squeeze(-1)  # (B*D, 64)
        x = x.reshape(B, D, -1)
        return self.dropout(self.proj(x))
```

## 4. iTransformer 完整实现

```python
class iTransformer(nn.Module):
    def __init__(self, n_vars, seq_len, pred_len, d_model=512,
                 n_heads=8, n_layers=3, d_ff=2048, dropout=0.1,
                 use_variate_attention=True):
        super().__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        self.d_model = d_model

        # Token 嵌入（每个变量 -> 一个 token）
        self.embedding = VariableTokenEmbedding(seq_len, d_model, dropout)

        # 变量位置编码（编码变量之间的关系）
        self.var_pos_encoding = nn.Parameter(
            torch.randn(1, n_vars, d_model) * 0.02
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # 预测头：每个变量 token 预测未来 pred_len 步
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, pred_len)
        )

    def forward(self, x, return_attention=False):
        # x: (B, L, D)
        B, L, D = x.shape

        # 1. 变量 Token 嵌入
        tokens = self.embedding(x)  # (B, D, d_model)

        # 2. 添加变量位置编码
        tokens = tokens + self.var_pos_encoding[:, :D, :]

        # 3. Transformer 编码（变量间注意力）
        encoded = self.encoder(tokens)  # (B, D, d_model)

        # 4. 预测
        pred = self.head(encoded)  # (B, D, pred_len)
        pred = pred.transpose(1, 2)  # (B, pred_len, D)

        return pred
```

## 5. 变量注意力的解读

iTransformer 的自注意力矩阵 $(D \times D)$ 揭示了变量间的关系：

```
注意力矩阵 (D=5):
         温度  湿度  风速  压强  降水
温度    [0.3  0.25 0.1  0.2  0.15]  ← 温度与湿度、压强相关
湿度    [0.2  0.35 0.1  0.15 0.2]  ← 湿度与降水相关
风速    [0.1  0.1  0.4  0.3  0.1]  ← 风速与压强相关
压强    [0.2  0.15 0.3  0.25 0.1]
降水    [0.15 0.2  0.1  0.1  0.45] ← 降水与湿度强相关
```

## 6. 通道独立 vs 通道混合 vs iTransformer

| 方法 | 通道策略 | Token 定义 | 建模能力 | 复杂度 |
|------|---------|-----------|---------|-------|
| PatchTST | 独立 | 补丁 | 时间模式 | $O(D \cdot (L/p)^2)$ |
| 标准 Transformer | 混合 | 时间步 | 时间+变量 | $O((D \cdot L)^2)$ |
| iTransformer | 混合 | 变量 | 变量关系+时序 | $O(D^2)$ |

## 7. 完整训练示例

```python
# ETT 数据集配置
model = iTransformer(
    n_vars=7,
    seq_len=96,
    pred_len=96,
    d_model=512,
    n_heads=8,
    n_layers=3,
    d_ff=2048,
    dropout=0.1
)

# 输入: (batch=32, seq_len=96, n_vars=7)
x = torch.randn(32, 96, 7)
pred = model(x)  # (32, 96, 7)

# 参数量分析
total = sum(p.numel() for p in model.parameters())
print(f'参数量: {total:,}')  # ~6M

# 与 PatchTST 对比
from patchtst import PatchTST
patchtst = PatchTST(n_vars=7, seq_len=96, pred_len=96,
                     patch_len=16, stride=8, d_model=128,
                     n_heads=4, n_layers=3)
patch_total = sum(p.numel() for p in patchtst.parameters())
print(f'PatchTST 参数量: {patch_total:,}')  # ~1M
```

## 8. 适用场景分析

**iTransformer 适合的场景：**
- 变量间存在强相关性（如交通流量多传感器、金融多指标）
- 变量数 $D$ 不太多（$D < 100$）
- 需要建模跨变量的因果关系

**不太适合的场景：**
- 变量间独立（此时 PatchTST 的通道独立更合适）
- 变量数极多（$D > 1000$，注意力矩阵过大）
- 单变量预测（退化为 MLP）

---

**要点总结：**
- iTransformer 的核心创新是将变量作为 token，将序列作为特征
- 注意力矩阵变为 $D \times D$，复杂度与序列长度无关
- 自然建模多变量间的时间依赖关系
- 与 PatchTST 互补：一个关注时间模式，一个关注变量关系
