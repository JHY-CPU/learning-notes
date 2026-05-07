# 1_Matrix Factorization 矩阵分解

## 1. 核心思想

矩阵分解将稀疏的用户-物品评分矩阵 $R \in \mathbb{R}^{m \times n}$ 分解为两个低秩矩阵的乘积：

$$R \approx P \cdot Q^T$$

其中 $P \in \mathbb{R}^{m \times k}$ 为用户隐因子矩阵，$Q \in \mathbb{R}^{n \times k}$ 为物品隐因子矩阵，$k \ll \min(m, n)$。

```
评分矩阵 R (m×n)    用户矩阵 P (m×k)   物品矩阵 Q (n×k)

[ 5  ?  3  ? ]       [p₁₁ p₁₂]         [q₁₁ q₁₂]
[ ?  4  ?  2 ]  ≈    [p₂₁ p₂₂]    ×    [q₂₁ q₂₂]ᵀ
[ 4  ?  ?  5 ]       [p₃₁ p₃₂]         [q₃₁ q₃₂]
[ ?  3  4  ? ]       [p₄₁ p₄₂]         [q₄₁ q₄₂]

预测: r̂ᵢⱼ = pᵢ · qⱼ = Σₖ pᵢₖ · qⱼₖ
```

## 2. 基础 MF 模型

### 2.1 损失函数

**均方误差损失：**
$$L = \sum_{(i,j) \in \text{observed}} (r_{ij} - p_i^T q_j)^2 + \lambda(\|p_i\|^2 + \|q_j\|^2)$$

```python
import numpy as np

class MatrixFactorization:
    """基础矩阵分解"""
    def __init__(self, n_users, n_items, n_factors=50, lr=0.01,
                 reg=0.02, n_epochs=100):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        # 随机初始化
        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))

    def fit(self, ratings):
        """
        ratings: list of (user_id, item_id, rating)
        """
        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)
            total_loss = 0

            for u, i, r in ratings:
                # 预测
                pred = np.dot(self.P[u], self.Q[i])
                error = r - pred

                # SGD 更新
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

                total_loss += error ** 2

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}, Loss: {total_loss / len(ratings):.4f}')

    def predict(self, user_id, item_id):
        return np.dot(self.P[user_id], self.Q[item_id])
```

## 3. SVD++ 与偏差项

基础 MF 忽略了用户和物品的固有偏差。SVD++ 加入偏差项：

$$\hat{r}_{ij} = \mu + b_i + b_j + p_i^T q_j$$

- $\mu$：全局平均评分
- $b_i$：用户 $i$ 的评分偏差（有人偏高，有人偏低）
- $b_j$：物品 $j$ 的评分偏差（有的电影普遍高分）

```python
class SVDPlusPlus:
    """SVD++ 带偏差项的矩阵分解"""
    def __init__(self, n_users, n_items, n_factors=50, lr=0.005,
                 reg=0.02, n_epochs=100):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))
        self.bu = np.zeros(n_users)  # 用户偏差
        self.bi = np.zeros(n_items)  # 物品偏差
        self.mu = 0                   # 全局均值

    def fit(self, ratings):
        self.mu = np.mean([r for _, _, r in ratings])

        for epoch in range(self.n_epochs):
            np.random.shuffle(ratings)
            total_loss = 0

            for u, i, r in ratings:
                pred = self.mu + self.bu[u] + self.bi[i] + \
                       np.dot(self.P[u], self.Q[i])
                error = r - pred

                # 更新偏差
                self.bu[u] += self.lr * (error - self.reg * self.bu[u])
                self.bi[i] += self.lr * (error - self.reg * self.bi[i])

                # 更新隐因子
                self.P[u] += self.lr * (error * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (error * self.P[u] - self.reg * self.Q[i])

                total_loss += error ** 2

            if (epoch + 1) % 10 == 0:
                rmse = np.sqrt(total_loss / len(ratings))
                print(f'Epoch {epoch+1}, RMSE: {rmse:.4f}')
```

## 4. ALS (Alternating Least Squares)

ALS 交替固定 P 或 Q，求解另一个：

```
固定 Q，求 P:  pᵢ = (Q_I^T Q_I + λI)^{-1} Q_I^T r_I
固定 P，求 Q:  qⱼ = (P_J^T P_J + λI)^{-1} P_J^T r_J
```

**优势：** 每步都是凸优化，可并行化。

```python
class ALS:
    """交替最小二乘矩阵分解"""
    def __init__(self, n_users, n_items, n_factors=50, reg=0.1, n_epochs=20):
        self.n_factors = n_factors
        self.reg = reg
        self.n_epochs = n_epochs

        self.P = np.random.normal(0, 0.1, (n_users, n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, n_factors))

    def fit(self, ratings_matrix):
        """
        ratings_matrix: (n_users, n_items) 稀疏矩阵
        """
        mask = ~np.isnan(ratings_matrix)

        for epoch in range(self.n_epochs):
            # 固定 Q，更新 P
            for i in range(self.P.shape[0]):
                rated_items = mask[i]
                if rated_items.sum() == 0:
                    continue
                Q_I = self.Q[rated_items]
                r_I = ratings_matrix[i, rated_items]
                # p_i = (Q_I^T Q_I + λI)^{-1} Q_I^T r_I
                A = Q_I.T @ Q_I + self.reg * np.eye(self.n_factors)
                b = Q_I.T @ r_I
                self.P[i] = np.linalg.solve(A, b)

            # 固定 P，更新 Q
            for j in range(self.Q.shape[0]):
                rated_users = mask[:, j]
                if rated_users.sum() == 0:
                    continue
                P_J = self.P[rated_users]
                r_J = ratings_matrix[rated_users, j]
                A = P_J.T @ P_J + self.reg * np.eye(self.n_factors)
                b = P_J.T @ r_J
                self.Q[j] = np.linalg.solve(A, b)
```

## 5. PyTorch 实现

```python
import torch
import torch.nn as nn

class MFModel(nn.Module):
    """PyTorch 矩阵分解模型"""
    def __init__(self, n_users, n_items, n_factors, use_bias=True):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, n_factors)
        self.item_embedding = nn.Embedding(n_items, n_factors)
        self.use_bias = use_bias

        if use_bias:
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)
            self.global_bias = nn.Parameter(torch.zeros(1))

        # 初始化
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)   # (B, k)
        item_emb = self.item_embedding(item_ids)   # (B, k)

        pred = (user_emb * item_emb).sum(dim=-1, keepdim=True)  # (B, 1)

        if self.use_bias:
            pred += self.user_bias(user_ids) + self.item_bias(item_ids) + self.global_bias

        return pred.squeeze(-1)

# 训练
model = MFModel(n_users=10000, n_items=5000, n_factors=64, use_bias=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
criterion = nn.MSELoss()

for epoch in range(50):
    model.train()
    for user_batch, item_batch, rating_batch in train_loader:
        pred = model(user_batch, item_batch)
        loss = criterion(pred, rating_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## 6. 隐因子数选择

| k (隐因子数) | 表达能力 | 过拟合风险 | 计算成本 |
|-------------|---------|-----------|---------|
| 10-30       | 弱      | 低        | 低      |
| 50-100      | 中      | 中        | 中      |
| 200+        | 强      | 高        | 高      |

**经验法则：** 通过验证集 AUC/NDCG 选择最优 k，通常 32-128 足够。

---

**要点总结：**
- 矩阵分解将评分矩阵分解为低秩用户/物品表示
- 偏差项 (SVD++) 捕捉用户和物品的固有偏差
- ALS 适合大规模分布式计算（Spark MLlib 实现）
- MF 是现代推荐系统的基石，深度学习模型大多在此基础上扩展
