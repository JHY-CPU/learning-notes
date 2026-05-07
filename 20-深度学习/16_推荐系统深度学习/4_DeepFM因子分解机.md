# 4_DeepFM 因子分解机

## 1. 概述

DeepFM (Huawei, 2017) 将 FM (Factorization Machine) 和 DNN 结合，共享嵌入层，同时建模低阶和高阶特征交互。

```
输入特征
    │
  嵌入层 (共享)
   / \
  FM   DNN
  │     │
  二阶  高阶
  交叉  交叉
   \   /
    ⊕
   输出
```

## 2. FM 回顾

因子分解机通过隐向量内积建模二阶特征交互：

$$\hat{y}_{FM} = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j$$

- $w_0$：全局偏置
- $w_i$：一阶特征权重
- $\langle v_i, v_j \rangle$：特征 $i$ 和 $j$ 的交叉系数（通过隐向量内积）

**高效计算技巧（避免 $O(n^2)$ 遍历）：**

$$\sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j = \frac{1}{2}\left[\left\|\sum_{i=1}^{n} v_i x_i\right\|^2 - \sum_{i=1}^{n}\|v_i x_i\|^2\right]$$

```python
import torch
import torch.nn as nn

class FMLayer(nn.Module):
    """FM 二阶交叉层"""
    def __init__(self, n_features, n_factors):
        super().__init__()
        self.n_features = n_features
        self.n_factors = n_factors

        # 一阶权重
        self.linear = nn.Linear(n_features, 1)
        # 隐向量
        self.V = nn.Parameter(torch.randn(n_features, n_factors) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # x: (B, n_features) 稀疏或稠密特征
        # 一阶项
        linear_part = self.linear(x) + self.bias

        # 二阶交叉项 (高效计算)
        # (Σ v_i * x_i)^2
        embed = x @ self.V  # (B, k)
        square_of_sum = embed.pow(2)  # (B, k)
        # Σ (v_i * x_i)^2
        sum_of_square = (x.pow(2) @ self.V.pow(2))  # (B, k)

        interaction = 0.5 * (square_of_sum - sum_of_square).sum(dim=1, keepdim=True)

        return linear_part + interaction
```

## 3. DeepFM 模型

```python
class DeepFM(nn.Module):
    """DeepFM: FM + DNN 共享嵌入"""
    def __init__(self, field_dims, embed_dim=16, mlp_dims=[256, 128, 64],
                 dropout=0.2):
        """
        field_dims: list, 每个字段的特征数量
            如 [10000, 5000, 200, 100] 表示 4 个字段
        """
        super().__init__()
        self.n_fields = len(field_dims)
        self.embed_dim = embed_dim

        # 每个字段一个嵌入表（FM 和 DNN 共享）
        self.embeddings = nn.ModuleList([
            nn.Embedding(dim, embed_dim) for dim in field_dims
        ])

        # FM 一阶权重
        self.linear_embeddings = nn.ModuleList([
            nn.Embedding(dim, 1) for dim in field_dims
        ])

        # DNN 部分
        dnn_input_dim = self.n_fields * embed_dim
        layers = []
        for mlp_dim in mlp_dims:
            layers.extend([
                nn.Linear(dnn_input_dim, mlp_dim),
                nn.ReLU(),
                nn.BatchNorm1d(mlp_dim),
                nn.Dropout(dropout)
            ])
            dnn_input_dim = mlp_dim
        self.mlp = nn.Sequential(*layers)
        self.mlp_output = nn.Linear(mlp_dims[-1], 1)

    def forward(self, x):
        """
        x: (B, n_fields) 每个字段的特征索引
        """
        # ========== FM 部分 ==========
        # 一阶
        linear_part = sum(
            self.linear_embeddings[i](x[:, i])
            for i in range(self.n_fields)
        )  # (B, 1)

        # 二阶：共享嵌入
        embeds = torch.stack([
            self.embeddings[i](x[:, i]) for i in range(self.n_fields)
        ], dim=1)  # (B, n_fields, embed_dim)

        # FM 交叉计算
        square_of_sum = embeds.sum(dim=1).pow(2).sum(dim=1, keepdim=True)
        sum_of_square = embeds.pow(2).sum(dim=1).sum(dim=1, keepdim=True)
        fm_part = linear_part + 0.5 * (square_of_sum - sum_of_square)

        # ========== DNN 部分 ==========
        dnn_input = embeds.view(x.size(0), -1)  # (B, n_fields * embed_dim)
        dnn_output = self.mlp_output(self.mlp(dnn_input))  # (B, 1)

        # ========== 合并 ==========
        output = fm_part + dnn_output
        return torch.sigmoid(output.squeeze(-1))

# 使用示例
model = DeepFM(
    field_dims=[10000, 5000, 200, 100, 50],  # 5个字段
    embed_dim=16,
    mlp_dims=[256, 128, 64]
)

# 输入: 每个字段的特征 ID
x = torch.stack([
    torch.randint(0, 10000, (32,)),
    torch.randint(0, 5000, (32,)),
    torch.randint(0, 200, (32,)),
    torch.randint(0, 100, (32,)),
    torch.randint(0, 50, (32,)),
], dim=1)  # (32, 5)

pred = model(x)  # (32,)
```

## 4. DeepFM vs Wide & Deep

| 特性 | Wide & Deep | DeepFM |
|------|------------|--------|
| Wide 部分 | 线性模型 + 手动交叉特征 | FM（自动二阶交叉） |
| 嵌入层 | Wide 和 Deep 独立 | 共享嵌入 |
| 特征工程 | 需要手动设计交叉 | 自动交叉 |
| 实现复杂度 | 需要两套特征处理 | 统一特征输入 |

**DeepFM 的优势：**
- 不需要手动设计交叉特征
- FM 和 DNN 共享嵌入，更高效
- 统一的输入格式，工程更简单

## 5. FM 的交叉阶数扩展

```python
class HigherOrderFM(nn.Module):
    """高阶 FM：通过分片 (FwL) 实现高阶交叉"""
    def __init__(self, n_features, n_factors, order=3):
        super().__init__()
        self.order = order
        # 每阶的隐向量
        self.V = nn.ParameterList([
            nn.Parameter(torch.randn(n_features, n_factors) * 0.01)
            for _ in range(order)
        ])

    def forward(self, x):
        # 通过递归嵌套实现高阶交叉
        result = torch.zeros(x.size(0), 1, device=x.device)
        for order in range(1, self.order + 1):
            # 计算 order 阶交叉
            embed = x @ self.V[order - 1]
            result += embed.pow(order).sum(dim=1, keepdim=True)
        return result
```

## 6. 实战技巧

```python
# 1. 学习率设置
optimizer = torch.optim.Adam([
    {'params': model.embeddings.parameters(), 'lr': 1e-3},
    {'params': model.mlp.parameters(), 'lr': 1e-3},
    {'params': model.linear_embeddings.parameters(), 'lr': 1e-2},
])

# 2. 嵌入正则化
def embedding_regularization(model, reg=1e-5):
    reg_loss = 0
    for emb in model.embeddings:
        reg_loss += emb.weight.norm(2)
    return reg * reg_loss

# 3. 训练循环
for epoch in range(epochs):
    model.train()
    for x_batch, y_batch in train_loader:
        pred = model(x_batch)
        loss = criterion(pred, y_batch.float())
        loss += embedding_regularization(model)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
```

---

**要点总结：**
- DeepFM 通过 FM 自动建模二阶特征交叉，DNN 建模高阶交叉
- 共享嵌入层是其关键设计，减少了参数和计算量
- FM 的高效计算技巧使其二阶交叉的时间复杂度降为 $O(nk)$
- DeepFM 因其简洁有效，成为 CTR 预测的经典基线模型
