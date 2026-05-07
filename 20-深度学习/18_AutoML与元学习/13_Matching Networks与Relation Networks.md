# 13_Matching Networks 与 Relation Networks

## 1. Matching Networks

### 1.1 核心思想

**Matching Networks (Vinyals et al., NeurIPS 2016)** 将少样本分类建模为**注意力加权的最近邻**：

$$P(y|\mathbf{x}, S) = \sum_{(\mathbf{x}_i, y_i) \in S} a(\mathbf{x}, \mathbf{x}_i) \cdot \mathbf{y}_i$$

其中注意力权重 $a(\mathbf{x}, \mathbf{x}_i)$ 由余弦相似度的 softmax 决定：

$$a(\mathbf{x}, \mathbf{x}_i) = \frac{\exp(\text{cos}(f(\mathbf{x}), g(\mathbf{x}_i)))}{\sum_{j} \exp(\text{cos}(f(\mathbf{x}), g(\mathbf{x}_j)))}$$

### 1.2 关键设计

- $f$ 和 $g$ 可以是不同的嵌入函数（非对称）
- 支持**全上下文嵌入 (Full Context Embeddings, FCE)**：每个 support 样本的嵌入依赖整个 support set

```python
class MatchingNetwork(nn.Module):
    def __init__(self, encoder, use_fce=False):
        super().__init__()
        self.encoder = encoder
        self.use_fce = use_fce
        
        if use_fce:
            # Full Context Embeddings: LSTM处理序列
            self.fce_lstm = nn.LSTM(
                input_size=encoder.output_dim,
                hidden_size=encoder.output_dim,
                bidirectional=True,
                batch_first=True
            )
    
    def forward(self, support_x, support_y, query_x, n_way):
        # 编码
        support_emb = self.encoder(support_x)  # (n_way * k_shot, d)
        query_emb = self.encoder(query_x)       # (n_query, d)
        
        # Full Context Embeddings
        if self.use_fce:
            # Bidirectional LSTM处理support序列
            support_emb_seq = support_emb.unsqueeze(0)  # (1, N, d)
            fce_out, _ = self.fce_lstm(support_emb_seq)
            support_emb = fce_out.squeeze(0) + support_emb  # 残差连接
        
        # 注意力权重（余弦相似度的softmax）
        # query_emb: (n_query, d), support_emb: (N, d)
        query_norm = F.normalize(query_emb, dim=-1)
        support_norm = F.normalize(support_emb, dim=-1)
        
        # 相似度矩阵
        similarity = torch.mm(query_norm, support_norm.t())  # (n_query, N)
        attention = F.softmax(similarity, dim=-1)  # (n_query, N)
        
        # 加权投票
        one_hot_y = F.one_hot(support_y, n_way).float()  # (N, n_way)
        logits = torch.mm(attention, one_hot_y)  # (n_query, n_way)
        
        return logits
```

### 1.3 FCE 的作用

```
无FCE: 每个support样本独立编码 → 固定向量
有FCE: support样本通过LSTM看到彼此 → 上下文感知的向量

效果: 当support set有共同模式时，FCE能捕获全局信息
```

## 2. Relation Network

### 2.1 核心思想

**Relation Network (Sung et al., CVPR 2018)** 用**神经网络**替代固定的距离度量（如欧氏距离）：

$$r_{ij} = g_\phi(f_\theta(\mathbf{x}_i), f_\theta(\mathbf{x}_j))$$

其中 $f_\theta$ 是嵌入网络，$g_\phi$ 是**关系模块**（学出来距离函数）。

### 2.2 架构

```
Support样本 → 嵌入网络 f_θ → 嵌入向量
                                    ↓
                          拼接每个(support, query)对
                                    ↓
                          关系模块 g_φ → 关系分数 [0, 1]
                                    ↓
                          最高分数的类 = 预测
```

```python
class RelationNetwork(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=8):
        super().__init__()
        # 嵌入网络
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, embed_dim, 3, padding=1), nn.BatchNorm2d(embed_dim), nn.ReLU(), nn.MaxPool2d(2),
        )
        
        # 关系模块：输入拼接的特征对，输出关系分数
        self.relation_module = nn.Sequential(
            nn.Conv2d(embed_dim * 2, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 8), nn.ReLU(),
            nn.Linear(8, 1), nn.Sigmoid(),  # 关系分数 [0, 1]
        )
    
    def forward(self, support_x, support_y, query_x, n_way):
        # 编码
        support_emb = self.encoder(support_x)  # (N, d, h, w)
        query_emb = self.encoder(query_x)      # (n_query, d, h, w)
        
        N = support_emb.shape[0]
        Q = query_emb.shape[0]
        
        # 构建所有(support, query)对
        # 扩展维度以配对
        support_exp = support_emb.unsqueeze(0).expand(Q, -1, -1, -1, -1)  # (Q, N, d, h, w)
        query_exp = query_emb.unsqueeze(1).expand(-1, N, -1, -1, -1)      # (Q, N, d, h, w)
        
        # 拼接
        pairs = torch.cat([support_exp, query_exp], dim=2)  # (Q, N, 2d, h, w)
        pairs = pairs.reshape(Q * N, -1, pairs.shape[-2], pairs.shape[-1])
        
        # 关系分数
        relations = self.relation_module(pairs)  # (Q * N, 1)
        relations = relations.reshape(Q, N)       # (Q, N)
        
        # 按类别聚合（平均关系分数）
        logits = torch.zeros(Q, n_way)
        for k in range(n_way):
            mask = (support_y == k)
            logits[:, k] = relations[:, mask].mean(dim=-1)
        
        return logits
```

## 3. 方法对比

| 特性 | Matching Net | Relation Net |
|------|-------------|-------------|
| 距离度量 | 余弦相似度 | 学习的神经网络 |
| 上下文嵌入 | FCE (LSTM) | 无 |
| 输出 | 加权投票 | 关系分数 |
| 计算复杂度 | $O(N)$ | $O(N \times Q)$ |
| 灵活性 | 低 | 高 |

## 4. 与 Prototypical Networks 对比

| 方法 | 原型计算 | 分类方式 | 特点 |
|------|----------|----------|------|
| ProtoNet | 均值嵌入 | 距离比较 | 简单高效 |
| Matching Net | 逐样本 | 注意力加权 | 上下文感知 |
| Relation Net | 无原型 | 关系网络 | 学习距离 |

### 4.1 适用场景

- **ProtoNet**：快速基线，原型有意义时效果好
- **Matching Net**：support set 有复杂模式时
- **Relation Net**：需要复杂距离度量时

## 5. 注意力变体

### 5.1 Soft k-NN

```python
def soft_knn(query_emb, support_emb, support_y, k=3, n_way=5):
    """软K近邻"""
    dists = torch.cdist(query_emb, support_emb)  # (Q, N)
    _, knn_idx = dists.topk(k, largest=False, dim=-1)  # (Q, k)
    
    knn_labels = support_y[knn_idx]  # (Q, k)
    knn_dists = dists.gather(1, knn_idx)  # (Q, k)
    
    # 距离倒数权重
    weights = 1.0 / (knn_dists + 1e-8)
    weights = F.softmax(weights, dim=-1)
    
    # 加权投票
    logits = torch.zeros(len(query_emb), n_way)
    for i in range(n_way):
        mask = (knn_labels == i).float()
        logits[:, i] = (weights * mask).sum(dim=-1)
    
    return logits
```

## 6. 总结

| 方法 | 年份 | 核心贡献 | 5-way 1-shot |
|------|------|----------|--------------|
| Matching Net | 2016 | 注意力加权最近邻 | ~43.6% |
| ProtoNet | 2017 | 类原型 | ~49.4% |
| Relation Net | 2018 | 学习距离度量 | ~50.4% |

---

**关键要点**：
1. Matching Networks 用注意力机制替代硬最近邻，引入上下文嵌入
2. Relation Network 用神经网络学习距离度量，比固定度量更灵活
3. ProtoNet 是最简单有效的基于度量的方法，通常作为基线
4. 基于度量的方法共同目标：学习好的嵌入空间，使得同类样本距离近
