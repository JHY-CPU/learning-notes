# 5_DIN 深度兴趣网络

## 1. 概述

DIN (Deep Interest Network, Alibaba, 2018) 针对电商推荐的序列行为建模，核心创新是**目标注意力 (Target Attention)**：根据当前候选物品，对用户历史行为进行自适应加权。

**动机：** 用户兴趣是多样的，不同候选物品应该激活用户历史中不同的行为。

```
用户历史: [球鞋, 鼠标, T恤, 键盘, 耳机]

候选 = 耳机:
  注意力权重: [0.05, 0.3, 0.02, 0.4, 0.23]  → 键盘、鼠标更重要

候选 = T恤:
  注意力权重: [0.35, 0.02, 0.4, 0.01, 0.22]  → 球鞋、T恤更重要
```

## 2. 目标注意力机制

DIN 的注意力以候选物品 (Ad) 为 Query，用户历史行为为 Key/Value：

```python
import torch
import torch.nn as nn

class TargetAttention(nn.Module):
    """DIN 目标注意力：根据候选物品对历史行为加权"""
    def __init__(self, embed_dim):
        super().__init__()
        # 注意力网络：输入 [用户行为, 候选物品, 用户行为-候选物品, 用户行为*候选物品]
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim * 4, 80),
            nn.PReLU(),
            nn.Linear(80, 40),
            nn.PReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, user_behavior_seq, candidate, mask=None):
        """
        user_behavior_seq: (B, L, D) 用户历史行为嵌入
        candidate: (B, D) 当前候选物品嵌入
        mask: (B, L) 历史行为的掩码
        """
        B, L, D = user_behavior_seq.shape

        # 扩展候选维度
        candidate = candidate.unsqueeze(1).expand(-1, L, -1)  # (B, L, D)

        # 构建注意力输入特征
        attention_input = torch.cat([
            user_behavior_seq,           # 历史行为
            candidate,                   # 候选物品
            user_behavior_seq - candidate,  # 差异
            user_behavior_seq * candidate,  # 交互
        ], dim=-1)  # (B, L, 4*D)

        # 计算注意力分数
        scores = self.attention_net(attention_input).squeeze(-1)  # (B, L)

        # 掩码处理
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax 注意力权重
        weights = torch.softmax(scores, dim=-1)  # (B, L)

        # 加权求和（用户兴趣表示）
        user_interest = torch.bmm(weights.unsqueeze(1), user_behavior_seq)
        user_interest = user_interest.squeeze(1)  # (B, D)

        return user_interest, weights
```

**为什么用 4 种输入特征：**

| 输入 | 含义 |
|------|------|
| 行为嵌入 | 历史物品本身 |
| 候选嵌入 | 当前要预测的物品 |
| 差 (行为-候选) | 相对差异 |
| 积 (行为*候选) | 元素级交互 |

## 3. DIN 完整模型

```python
class DIN(nn.Module):
    """Deep Interest Network"""
    def __init__(self, user_feature_dims, item_feature_dims,
                 context_feature_dims, embed_dim=8,
                 mlp_dims=[200, 80]):
        super().__init__()
        self.embed_dim = embed_dim

        # 用户行为嵌入（用于历史序列）
        self.item_embedding = nn.Embedding(item_feature_dims['item_id'], embed_dim)

        # 其他特征嵌入
        self.user_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in user_feature_dims.items()
            if name != 'behavior_seq'
        })
        self.context_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in context_feature_dims.items()
        })

        # 候选物品嵌入
        self.candidate_embeddings = nn.ModuleDict({
            name: nn.Embedding(dim, embed_dim)
            for name, dim in item_feature_dims.items()
        })

        # 目标注意力
        self.attention = TargetAttention(embed_dim)

        # DNN
        # 输入维度 = 用户特征 + 注意力输出 + 候选物品特征 + 上下文特征
        total_dim = (len(self.user_embeddings) + 1 +  # 用户 + 注意力输出
                     len(self.candidate_embeddings) +
                     len(self.context_embeddings)) * embed_dim

        layers = []
        prev_dim = total_dim
        for mlp_dim in mlp_dims:
            layers.extend([
                nn.Linear(prev_dim, mlp_dim),
                nn.PReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = mlp_dim
        self.mlp = nn.Sequential(*layers)
        self.output_layer = nn.Linear(mlp_dims[-1], 1)

    def forward(self, user_features, candidate_features, context_features,
                behavior_seq, behavior_mask):
        """
        user_features: dict of (B,) 用户画像特征
        candidate_features: dict of (B,) 候选物品特征
        context_features: dict of (B,) 上下文特征
        behavior_seq: (B, L) 用户历史行为物品 ID
        behavior_mask: (B, L) 有效行为掩码
        """
        # 1. 用户历史行为嵌入
        behavior_embeds = self.item_embedding(behavior_seq)  # (B, L, D)

        # 2. 候选物品嵌入
        candidate_embed = sum(
            self.candidate_embeddings[name](candidate_features[name])
            for name in self.candidate_embeddings
        )  # (B, D)

        # 3. 目标注意力：根据候选物品对历史行为加权
        user_interest, attn_weights = self.attention(
            behavior_embeds, candidate_embed, behavior_mask
        )

        # 4. 用户画像嵌入
        user_embeds = []
        for name, emb in self.user_embeddings.items():
            user_embeds.append(emb(user_features[name]))
        user_embed = torch.stack(user_embeds, dim=1).view(user_embeds[0].size(0), -1)

        # 5. 候选物品其他特征
        candidate_embeds = []
        for name, emb in self.candidate_embeddings.items():
            candidate_embeds.append(emb(candidate_features[name]))
        candidate_total = torch.stack(candidate_embeds, dim=1).view(
            candidate_embeds[0].size(0), -1
        )

        # 6. 上下文嵌入
        context_embeds = []
        for name, emb in self.context_embeddings.items():
            context_embeds.append(emb(context_features[name]))
        context_embed = torch.stack(context_embeds, dim=1).view(
            context_embeds[0].size(0), -1
        )

        # 7. 拼接所有特征
        all_features = torch.cat([
            user_embed, user_interest, candidate_total, context_embed
        ], dim=-1)

        # 8. DNN 预测
        hidden = self.mlp(all_features)
        output = torch.sigmoid(self.output_layer(hidden))
        return output.squeeze(-1), attn_weights
```

## 4. Dice 激活函数

DIN 提出了 Dice 激活函数，自适应调整 ReLU 的阈值：

$$\text{Dice}(x) = p(x) \cdot x + (1 - p(x)) \cdot \alpha x$$
$$p(x) = \sigma(\frac{x - E[x]}{\sqrt{Var[x] + \epsilon}})$$

```python
class Dice(nn.Module):
    """Dice 激活函数：自适应 PReLU"""
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(d_model))
        self.bn = nn.BatchNorm1d(d_model, affine=False)

    def forward(self, x):
        if x.dim() == 2:
            x_normed = self.bn(x)
        else:
            x_normed = self.bn(x.transpose(1, 2)).transpose(1, 2)

        p = torch.sigmoid(x_normed)
        return p * x + (1 - p) * self.alpha * x
```

## 5. DIN vs 普通 CTR 模型

| 特性 | 普通 DNN | DIN |
|------|---------|-----|
| 用户表示 | 静态（固定向量） | 动态（根据候选物品变化） |
| 历史行为 | 池化（平均/求和） | 注意力加权 |
| 兴趣多样性 | 无法区分 | 目标感知 |
| 效果 | baseline | 显著提升 |

## 6. 实战要点

```python
# 1. 行为序列长度截断
MAX_SEQ_LEN = 50  # 实际中通常截断到 20-100

# 2. 序列按时间排序（最新在前或在后，保持一致）
# 3. 掩码处理变长序列
# 4. 注意力计算可以缓存（离线计算用户兴趣向量）

# 5. 负采样
def negative_sampling(positive_items, n_items, n_neg=4):
    """随机负采样"""
    neg_items = []
    for _ in range(n_neg):
        neg = np.random.randint(0, n_items)
        while neg in positive_items:
            neg = np.random.randint(0, n_items)
        neg_items.append(neg)
    return neg_items
```

---

**要点总结：**
- DIN 通过目标注意力机制，根据候选物品动态调整用户历史行为的权重
- 注意力网络输入 4 种特征组合（行为、候选、差、积），增强表达能力
- Dice 激活函数自适应调整非线性阈值
- DIN 开创了序列行为建模 + 注意力机制的推荐范式
