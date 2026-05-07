# 12_Prototypical Networks

## 1. 核心思想

**Prototypical Networks (Snell et al., NeurIPS 2017)** 为每个类别计算一个**原型 (Prototype)**，通过查询样本到各原型的距离进行分类。

### 1.1 直觉

```
每类的所有 support 样本
    ↓ 编码器嵌入
    ↓ 取平均
该类的"原型"（代表向量）

新样本 → 编码器 → 与各原型比较距离 → 最近的类就是预测类
```

### 1.2 数学形式

对类别 $k$，原型定义为该类 support 样本嵌入的均值：

$$\mathbf{c}_k = \frac{1}{|S_k|} \sum_{(\mathbf{x}_i, y_i) \in S_k} f_\theta(\mathbf{x}_i)$$

分类基于**负欧氏距离**的 softmax：

$$p(y = k | \mathbf{x}) = \frac{\exp(-d(f_\theta(\mathbf{x}), \mathbf{c}_k))}{\sum_{k'} \exp(-d(f_\theta(\mathbf{x}), \mathbf{c}_{k'}))}$$

其中 $d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|^2$ 是欧氏距离。

## 2. 完整实现

### 2.1 编码器

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_encoder(input_channels=1):
    """4层卷积编码器（标准few-shot backbone）"""
    return nn.Sequential(
        nn.Conv2d(input_channels, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(64, 64, 3, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(2),
        
        Flatten(),
    )
```

### 2.2 Prototypical Network

```python
class PrototypicalNetwork(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    
    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """计算每个类的原型"""
        prototypes = []
        for k in range(n_way):
            # 选出该类的嵌入
            mask = (support_labels == k)
            class_embeddings = support_embeddings[mask]
            # 均值作为原型
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)
        return torch.stack(prototypes)  # (n_way, embedding_dim)
    
    def forward(self, support_x, support_y, query_x, n_way):
        """
        support_x: (n_way * k_shot, C, H, W)
        support_y: (n_way * k_shot,)
        query_x: (n_query, C, H, W)
        返回: (n_query, n_way) 分类logits
        """
        # 编码
        support_emb = self.encoder(support_x)  # (n_way * k_shot, d)
        query_emb = self.encoder(query_x)      # (n_query, d)
        
        # 计算原型
        prototypes = self.compute_prototypes(support_emb, support_y, n_way)
        
        # 计算距离
        # query_emb: (n_query, d), prototypes: (n_way, d)
        dists = torch.cdist(query_emb, prototypes)  # (n_query, n_way)
        
        # 距离取负作为logits（距离越小，概率越大）
        logits = -dists
        
        return logits
```

### 2.3 训练

```python
def train_protonet(model, train_loader, optimizer, n_way=5, k_shot=1, q_query=15):
    """训练Prototypical Network"""
    model.train()
    total_loss = 0
    total_acc = 0
    n_episodes = 0
    
    for batch in train_loader:
        support_x, support_y, query_x, query_y = batch
        
        # 前向传播
        logits = model(support_x, support_y, query_x, n_way)
        
        # 损失
        loss = F.cross_entropy(logits, query_y)
        
        # 精度
        pred = logits.argmax(dim=-1)
        acc = (pred == query_y).float().mean()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        n_episodes += 1
    
    return total_loss / n_episodes, total_acc / n_episodes
```

## 3. 距离度量的选择

### 3.1 欧氏距离（默认）

$$d(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|^2_2$$

### 3.2 余弦距离

$$d(\mathbf{a}, \mathbf{b}) = 1 - \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \|\mathbf{b}\|}$$

```python
def cosine_distance(a, b):
    """余弦距离"""
    a_norm = F.normalize(a, dim=-1)
    b_norm = F.normalize(b, dim=-1)
    return 1 - torch.mm(a_norm, b_norm.t())
```

### 3.3 学习型距离

```python
class LearnedDistance(nn.Module):
    """学习型距离度量"""
    def __init__(self, embed_dim):
        super().__init__()
        self.fc = nn.Linear(embed_dim * 2, 1)
    
    def forward(self, a, b):
        # a: (n, d), b: (m, d)
        n, m = a.shape[0], b.shape[0]
        a_expand = a.unsqueeze(1).expand(n, m, -1)
        b_expand = b.unsqueeze(0).expand(n, m, -1)
        concat = torch.cat([a_expand, b_expand], dim=-1)
        return self.fc(concat).squeeze(-1)
```

## 4. 与 MAML 的对比

| 特性 | MAML | ProtoNet |
|------|------|----------|
| 范式 | 基于优化 | 基于度量 |
| 适应方式 | 梯度下降 | 原型计算 |
| 训练速度 | 慢（二层优化） | 快（端到端） |
| 适应灵活性 | 高 | 低（只分类） |
| 5-way 1-shot | ~48.7% | ~49.4% |
| 5-way 5-shot | ~63.1% | ~68.2% |

## 5. 扩展

### 5.1 Semi-ProtoNet

利用未标注数据改进原型估计：

```python
def semi_prototypes(model, labeled_x, labeled_y, unlabeled_x, n_way):
    """半监督原型计算"""
    labeled_emb = model.encoder(labeled_x)
    unlabeled_emb = model.encoder(unlabeled_x)
    
    # 初始原型（仅用标注数据）
    prototypes = compute_prototypes(labeled_emb, labeled_y, n_way)
    
    # 迭代：用当前原型预测未标注数据的标签 → 更新原型
    for _ in range(5):
        dists = torch.cdist(unlabeled_emb, prototypes)
        pseudo_labels = dists.argmin(dim=-1)
        
        for k in range(n_way):
            mask_l = (labeled_y == k)
            mask_u = (pseudo_labels == k)
            
            all_emb = torch.cat([labeled_emb[mask_l], unlabeled_emb[mask_u]], dim=0)
            prototypes[k] = all_emb.mean(dim=0)
    
    return prototypes
```

### 5.2 加权原型

根据嵌入质量给样本不同权重：

```python
def weighted_prototype(embeddings, confidence_scores):
    """加权原型"""
    weights = F.softmax(confidence_scores, dim=0)
    return (weights.unsqueeze(-1) * embeddings).sum(dim=0)
```

---

**关键要点**：
1. ProtoNet 通过计算类原型（均值嵌入）进行分类，简单高效
2. 欧氏距离是默认的距离度量，余弦距离在高维空间通常更好
3. 相比MAML，ProtoNet训练更快且无需二阶导数
4. ProtoNet 的性能高度依赖于编码器的嵌入质量
