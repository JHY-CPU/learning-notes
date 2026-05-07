# 2_Deep Crossing 与嵌入层

## 1. 概述

Deep Crossing (Microsoft, 2016) 是最早将深度学习应用于推荐系统的模型之一。核心贡献：
1. **Embedding 层**：将稀疏 ID 特征映射为稠密向量
2. **残差连接**：堆叠多层全连接网络，通过残差连接训练深层模型

## 2. 稀疏特征与嵌入

推荐系统中大量特征是稀疏的类别特征（用户ID、物品ID、类别等），需要嵌入层转换：

```
用户特征 (one-hot, 维度=10000):
  [0, 0, ..., 1, ..., 0]  →  Embedding  →  [0.23, -0.15, 0.78, ...]  (64维)

物品类别 (multi-hot, 维度=100):
  [0, 1, 0, 1, 0, ..., 0]  →  Embedding(平均)  →  [0.41, -0.08, ...]  (32维)
```

```python
import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    """推荐系统嵌入层：处理稀疏和稠密特征"""
    def __init__(self, sparse_feature_dims, dense_feature_dim,
                 embedding_dim=32):
        super().__init__()
        self.sparse_embeddings = nn.ModuleDict()
        self.sparse_feature_dims = sparse_feature_dims

        for name, vocab_size in sparse_feature_dims.items():
            self.sparse_embeddings[name] = nn.Embedding(
                vocab_size, embedding_dim
            )

        # 稠密特征投影
        self.dense_proj = nn.Linear(dense_feature_dim, embedding_dim)

    def forward(self, sparse_inputs, dense_inputs):
        """
        sparse_inputs: dict of (B,) 离散特征 ID
        dense_inputs: (B, D_dense) 连续特征
        """
        embedded = []

        # 稀疏特征嵌入
        for name, input_ids in sparse_inputs.items():
            emb = self.sparse_embeddings[name](input_ids)  # (B, embed_dim)
            embedded.append(emb)

        # 稠密特征投影
        dense_emb = self.dense_proj(dense_inputs)  # (B, embed_dim)
        embedded.append(dense_emb)

        # 拼接所有特征
        return torch.cat(embedded, dim=-1)  # (B, total_embed_dim)

# 示例
embedding_layer = EmbeddingLayer(
    sparse_feature_dims={
        'user_id': 10000,
        'item_id': 5000,
        'category': 100,
        'city': 300,
    },
    dense_feature_dim=10,
    embedding_dim=32
)

sparse_inputs = {
    'user_id': torch.randint(0, 10000, (32,)),
    'item_id': torch.randint(0, 5000, (32,)),
    'category': torch.randint(0, 100, (32,)),
    'city': torch.randint(0, 300, (32,)),
}
dense_inputs = torch.randn(32, 10)

# 输出维度: 4 * 32 + 32 = 160
features = embedding_layer(sparse_inputs, dense_inputs)
print(features.shape)  # torch.Size([32, 160])
```

## 3. Embedding 维度选择

| 特征类型 | 建议维度 | 说明 |
|---------|---------|------|
| 用户/物品 ID | 16-128 | 取决于 ID 数量 |
| 高基数类别 | 8-32 | 如城市、设备类型 |
| 低基数类别 | 4-16 | 如性别、操作系统 |
| 连续特征 | 与 embed 维度一致 | 线性投影 |

**经验公式：** $d_{emb} = \min(50, \text{round}(\text{vocab\_size}^{0.25}))$

## 4. Deep Crossing 模型

```python
class ResidualBlock(nn.Module):
    """残差块：用于深层全连接网络"""
    def __init__(self, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return self.layer_norm(residual + out)


class DeepCrossing(nn.Module):
    """Deep Crossing: 残差连接 DNN 推荐模型"""
    def __init__(self, sparse_feature_dims, dense_feature_dim,
                 embedding_dim=32, hidden_dims=[256, 128, 64],
                 dropout=0.1, output_dim=1):
        super().__init__()

        # 嵌入层
        self.embedding = EmbeddingLayer(
            sparse_feature_dims, dense_feature_dim, embedding_dim
        )

        # 计算总嵌入维度
        total_embed_dim = len(sparse_feature_dims) * embedding_dim + embedding_dim

        # 投影层（统一维度）
        self.proj = nn.Linear(total_embed_dim, hidden_dims[0])

        # 残差块堆叠
        self.res_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.res_blocks.append(
                ResidualBlock(hidden_dims[i], hidden_dims[i] * 4, dropout)
            )
            # 如果维度变化，添加投影
            if hidden_dims[i] != hidden_dims[i + 1]:
                self.res_blocks.append(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1])
                )

        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, sparse_inputs, dense_inputs):
        # 嵌入
        x = self.embedding(sparse_inputs, dense_inputs)

        # 投影
        x = self.proj(x)
        x = torch.relu(x)

        # 残差块
        for block in self.res_blocks:
            x = block(x)

        return torch.sigmoid(self.output_layer(x)).squeeze(-1)
```

## 5. Embedding 共享与预训练

```python
class SharedEmbedding(nn.Module):
    """跨任务共享嵌入"""
    def __init__(self, n_users, n_items, embed_dim, use_pretrained=True):
        super().__init__()
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)

        if use_pretrained:
            # 用 MF 预训练嵌入初始化
            self._load_pretrained()

    def _load_pretrained(self):
        """从 MF 模型加载预训练嵌入"""
        # mf_user_emb = np.load('mf_user_emb.npy')
        # self.user_embed.weight.data = torch.FloatTensor(mf_user_emb)
        pass

    def forward(self, user_ids, item_ids):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        return u, i
```

## 6. 多值特征处理

```python
class MultiValueEmbedding(nn.Module):
    """多值特征嵌入（如用户浏览过的物品列表）"""
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

    def forward(self, multi_hot_ids):
        """
        multi_hot_ids: (B, max_len) 变长列表（padding=0）
        """
        emb = self.embedding(multi_hot_ids)  # (B, max_len, embed_dim)

        # 掩码平均池化
        mask = (multi_hot_ids != 0).unsqueeze(-1).float()  # (B, max_len, 1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return pooled  # (B, embed_dim)
```

## 7. 从 MF 到 Deep Crossing

| 特性 | MF | Deep Crossing |
|------|-----|---------------|
| 特征类型 | 仅 ID | ID + 类别 + 连续 |
| 交互建模 | 内积（二阶） | DNN（高阶） |
| 网络深度 | 0 层 | 多层残差 |
| 非线性 | 无 | ReLU |
| 表达能力 | 有限 | 强 |

---

**要点总结：**
- Embedding 层是深度推荐系统的基础，将稀疏特征转为稠密表示
- Deep Crossing 首次将残差连接引入推荐系统的 DNN
- 嵌入维度需要根据特征基数和数据量精心选择
- 预训练嵌入（如用 MF）可以加速收敛并提升性能
