# 3_Wide & Deep 模型

## 1. 概述

Wide & Deep (Google, 2016) 是推荐系统里程碑式的模型，发表于 Google Play 应用推荐场景。其核心理念是同时利用**记忆 (Memorization)** 和**泛化 (Generalization)**。

```
        输入特征
       /         \
    Wide        Deep
  (记忆)       (泛化)
     |           |
  线性模型    Embedding + DNN
     |           |
     └────+─────┘
          |
       输出层
```

## 2. 记忆 vs 泛化

| 维度 | 记忆 (Wide) | 泛化 (Deep) |
|------|------------|-------------|
| 含义 | 学习特征间的直接共现关系 | 学习未见过的特征组合 |
| 模型 | 线性模型 + 交叉特征 | Embedding + 多层 DNN |
| 优势 | 精确匹配历史模式 | 发现新模式 |
| 劣势 | 无法泛化到新组合 | 可能遗忘已知模式 |

**例子：**
- **记忆：** "安装了 A 游戏的用户也会安装 B 游戏" — 直接共现
- **泛化：** "喜欢策略游戏的用户可能喜欢某款新策略游戏" — 从未见过但符合模式

## 3. 模型架构

```python
import torch
import torch.nn as nn

class WideAndDeep(nn.Module):
    def __init__(self, wide_feature_dim, deep_sparse_dims, deep_dense_dim,
                 embed_dim=32, hidden_dims=[256, 128, 64], dropout=0.1):
        super().__init__()

        # ====== Wide 部分（线性模型）======
        self.wide_linear = nn.Linear(wide_feature_dim, 1)

        # ====== Deep 部分 ======
        # 嵌入层
        self.deep_embeddings = nn.ModuleDict({
            name: nn.Embedding(vocab_size, embed_dim)
            for name, vocab_size in deep_sparse_dims.items()
        })

        # 稠密特征投影
        self.dense_proj = nn.Linear(deep_dense_dim, embed_dim)

        # Deep 部分总维度
        deep_input_dim = len(deep_sparse_dims) * embed_dim + embed_dim

        # DNN 层
        layers = []
        prev_dim = deep_input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(dropout)
            ])
            prev_dim = h_dim
        self.deep_network = nn.Sequential(*layers)
        self.deep_output = nn.Linear(hidden_dims[-1], 1)

        # 输出层
        self.output_bias = nn.Parameter(torch.zeros(1))

    def forward(self, wide_features, deep_sparse_inputs, deep_dense_inputs):
        """
        wide_features: (B, wide_dim) 交叉特征
        deep_sparse_inputs: dict of (B,)
        deep_dense_inputs: (B, dense_dim)
        """
        # Wide 部分
        wide_out = self.wide_linear(wide_features)  # (B, 1)

        # Deep 部分 - 嵌入
        deep_emb = []
        for name, ids in deep_sparse_inputs.items():
            deep_emb.append(self.deep_embeddings[name](ids))
        deep_emb.append(self.dense_proj(deep_dense_inputs))
        deep_input = torch.cat(deep_emb, dim=-1)

        # Deep 部分 - DNN
        deep_hidden = self.deep_network(deep_input)
        deep_out = self.deep_output(deep_hidden)  # (B, 1)

        # 合并输出
        combined = wide_out + deep_out + self.output_bias
        return torch.sigmoid(combined).squeeze(-1)
```

## 4. 交叉特征 (Cross Features)

Wide 部分的关键是手动设计交叉特征：

```python
def create_cross_features(df):
    """Wide 部分的交叉特征"""
    cross_features = {}

    # 二阶交叉
    cross_features['user_gender_x_item_category'] = (
        df['user_gender'].astype(str) + '_' + df['item_category'].astype(str)
    )
    cross_features['user_age_bucket_x_item_price_bucket'] = (
        df['user_age_bucket'].astype(str) + '_' + df['item_price_bucket'].astype(str)
    )
    cross_features['installed_app_x_current_app'] = (
        df['installed_app_id'].astype(str) + '_' + df['current_app_id'].astype(str)
    )

    # 独热编码交叉特征
    from sklearn.preprocessing import OneHotEncoder
    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    wide_features = encoder.fit_transform(
        pd.DataFrame(cross_features)
    )
    return wide_features
```

## 5. Google Play 的原始设计

```
Wide 部分:
  输入: 已安装应用 × 当前候选应用 的交叉特征
  模型: L1 正则化逻辑回归

Deep 部分:
  输入: 用户画像（国家、语言、年龄）+ 应用特征（类别、开发者）
  嵌入: 每个稀疏特征嵌入为 32 维
  DNN: 两层 ReLU（256 → 128 → 64）

训练:
  联合训练，反向传播同时更新 Wide 和 Deep 部分
  优化器: Follow-the-Regularized-Leader (FTRL) for Wide
           AdaGrad for Deep
```

## 6. 改进：Deep & Cross Network (DCN)

DCN 用交叉网络替代手动交叉特征：

```python
class CrossNetwork(nn.Module):
    """交叉网络：自动学习特征交叉"""
    def __init__(self, input_dim, n_layers):
        super().__init__()
        self.n_layers = n_layers
        self.W = nn.ModuleList([
            nn.Linear(input_dim, 1, bias=False) for _ in range(n_layers)
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim)) for _ in range(n_layers)
        ])

    def forward(self, x):
        """
        x_0: (B, D) 初始特征
        交叉: x_{l+1} = x_0 ⊙ (W_l · x_l + b_l) + x_l
        """
        x_0 = x
        for i in range(self.n_layers):
            x_w = self.W[i](x)  # (B, 1)
            x = x_0 * x_w + self.b[i] + x  # (B, D)
        return x


class DCNV2(nn.Module):
    """DCN-V2: 深度交叉网络"""
    def __init__(self, input_dim, cross_layers, hidden_dims):
        super().__init__()
        self.cross_net = CrossNetwork(input_dim, cross_layers)
        self.deep_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU()
        )
        self.output = nn.Linear(input_dim + hidden_dims[-1], 1)

    def forward(self, x):
        cross_out = self.cross_net(x)    # (B, D)
        deep_out = self.deep_net(x)      # (B, hidden)
        combined = torch.cat([cross_out, deep_out], dim=-1)
        return torch.sigmoid(self.output(combined)).squeeze(-1)
```

## 7. Wide & Deep vs DCN 对比

| 特性 | Wide & Deep | DCN / DCN-V2 |
|------|------------|-------------|
| 交叉特征 | 手动设计 | 自动学习 |
| 交叉阶数 | 受限于手动特征 | 显式控制层数 |
| 工程成本 | 高（特征工程） | 低（端到端） |
| 计算效率 | 快 | 交叉网络开销小 |
| 实际效果 | 依赖特征质量 | 更鲁棒 |

## 8. 实战使用

```python
# 模型定义
model = WideAndDeep(
    wide_feature_dim=1000,      # 交叉特征独热维度
    deep_sparse_dims={
        'user_id': 100000,
        'item_id': 50000,
        'category': 200,
        'tag': 1000,
    },
    deep_dense_dim=20,           # 连续特征维度
    embed_dim=32,
    hidden_dims=[256, 128, 64],
    dropout=0.2
)

# 输入
wide_features = torch.randn(32, 1000)
deep_sparse = {
    'user_id': torch.randint(0, 100000, (32,)),
    'item_id': torch.randint(0, 50000, (32,)),
    'category': torch.randint(0, 200, (32,)),
    'tag': torch.randint(0, 1000, (32,)),
}
deep_dense = torch.randn(32, 20)

# 前向传播
pred = model(wide_features, deep_sparse, deep_dense)
print(f'预测值: {pred.shape}')  # (32,)
```

---

**要点总结：**
- Wide & Deep 同时利用线性模型的记忆能力和 DNN 的泛化能力
- Wide 部分的关键是交叉特征设计，需要领域知识
- DCN 用自动交叉替代手动交叉，降低了特征工程成本
- 工业实践中，DCN-V2 因其自动交叉能力被广泛采用
