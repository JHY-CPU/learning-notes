# 3_GCN 图卷积网络：谱域到空域的简化

## 1. 图卷积的起源：谱图理论

### 1.1 从传统卷积到图卷积

传统 CNN 的卷积在频域中等价于频率域的乘法：

$$f * g = \mathcal{F}^{-1}(\mathcal{F}(f) \cdot \mathcal{F}(g))$$

在图上，我们用拉普拉斯矩阵的特征向量作为"图傅里叶基"，定义图卷积。

### 1.2 图傅里叶变换

拉普拉斯矩阵 $L = D - A$ 的特征分解：

$$L = U \Lambda U^T$$

- $U$ 的列 $u_0, u_1, \dots, u_{N-1}$ 是特征向量（图傅里叶基）
- $\Lambda = \text{diag}(\lambda_0, \dots, \lambda_{N-1})$ 是特征值（图频率）

**图傅里叶变换**：

$$\hat{x}_l = \sum_{i=1}^N u_l(i) x_i = u_l^T x$$

**逆变换**：

$$x = \sum_{l=0}^{N-1} \hat{x}_l u_l = U \hat{x}$$

### 1.3 谱图卷积

定义图卷积为：

$$x *_G g_\theta = U g_\theta(\Lambda) U^T x$$

其中 $g_\theta(\Lambda) = \text{diag}(g_\theta(\lambda_0), \dots, g_\theta(\lambda_{N-1}))$ 是可学习的谱滤波器。

**问题**：需要对 $L$ 做特征分解（$O(N^3)$），且 $U$ 是稠密矩阵——计算代价极高。

## 2. ChebNet：切比雪夫多项式近似

### 2.1 核心思想

Defferrard et al. (2016) 提出用 $K$ 阶切比雪夫多项式近似滤波器：

$$g_\theta(\Lambda) \approx \sum_{k=0}^{K} \theta_k T_k(\tilde{\Lambda})$$

其中 $\tilde{\Lambda} = \frac{2\Lambda}{\lambda_{\max}} - I$，$T_k$ 是切比雪夫多项式：

$$T_0(x) = 1, \quad T_1(x) = x, \quad T_k(x) = 2x T_{k-1}(x) - T_{k-2}(x)$$

**关键优势**：$T_k(\tilde{L})$ 可以递推计算，不需要特征分解！

$$y = \sum_{k=0}^{K} \theta_k T_k(\tilde{L}) x$$

其中 $\tilde{L} = \frac{2L}{\lambda_{\max}} - I$。

### 2.2 实现

```python
class ChebConv(nn.Module):
    def __init__(self, in_dim, out_dim, K):
        super().__init__()
        self.K = K
        self.theta = nn.Parameter(torch.randn(K + 1, in_dim, out_dim))

    def forward(self, x, L_norm):
        # L_norm = 2L/λ_max - I
        T = [torch.ones_like(x), L_norm @ x]  # T_0, T_1
        for k in range(2, self.K + 1):
            T_k = 2 * L_norm @ T[-1] - T[-2]
            T.append(T_k)

        out = sum(T[k] @ self.theta[k] for k in range(self.K + 1))
        return out
```

## 3. Kipf & Welling 简化（GCN）

### 3.1 K=1 的近似

Kipf & Welling (2017) 做了两个关键简化：

1. **$K=1$**（只用一阶近似），限制感受野为直接邻居
2. **$\lambda_{\max} \approx 2$**（对归一化拉普拉斯）

将 $\tilde{L} \approx L - I = -D^{-1/2} A D^{-1/2}$（归一化形式），得到：

$$g_\theta *_G x \approx \theta_0 x + \theta_1 D^{-1/2} A D^{-1/2} x$$

进一步令 $\theta = \theta_0 = -\theta_1$（减少参数，避免数值不稳定），得到**GCN 层公式**：

$$H^{(l+1)} = \sigma\left(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}\right)$$

其中：
- $\hat{A} = A + I_N$（添加自环）
- $\hat{D}_{ii} = \sum_j \hat{A}_{ij}$（新的度矩阵）
- $W^{(l)} \in \mathbb{R}^{d^{(l)} \times d^{(l+1)}}$ 是可学习权重

### 3.2 直观理解

GCN 的每一步相当于：
1. **添加自环**：节点保留自己的信息
2. **归一化**：按度数归一化，避免高节点度主导
3. **线性变换**：特征空间映射
4. **激活**：引入非线性

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

# ========== 从零实现 GCN ==========
class GCNFromScratch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.W1 = nn.Linear(in_dim, hidden_dim)
        self.W2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index, num_nodes):
        # 构建归一化邻接矩阵
        A = torch.zeros(num_nodes, num_nodes)
        A[edge_index[0], edge_index[1]] = 1.0
        A_hat = A + torch.eye(num_nodes)

        D_hat = torch.diag(A_hat.sum(1))
        D_inv_sqrt = torch.diag(1.0 / torch.sqrt(A_hat.sum(1)))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0

        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt  # 对称归一化

        # 两层 GCN
        h = F.relu(A_norm @ self.W1(x))
        h = A_norm @ self.W2(h)
        return h

# ========== 使用 PyG 的 GCN ==========
class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## 4. GCN 的传播规则推导

### 4.1 对称归一化的推导

$$\hat{A}_{\text{norm}} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2}$$

矩阵元素形式：

$$(\hat{A}_{\text{norm}})_{ij} = \frac{1}{\sqrt{\hat{d}_i \hat{d}_j}} \hat{A}_{ij}$$

其中 $\hat{d}_i = 1 + \deg(v_i)$（因为加了自环）。

**直觉**：高度数节点的贡献被"压制"——来自高度数邻居的消息权重小，来自低度数邻居的消息权重大。

### 4.2 右乘 vs 左乘

$$H^{(l+1)} = \hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)}$$

先做 $H^{(l)} W^{(l)}$（线性变换），再乘归一化邻接矩阵（邻居聚合）。等价写法：

$$H^{(l+1)} = \hat{D}^{-1/2} \hat{A} (\hat{D}^{-1/2} H^{(l)} W^{(l)})$$

## 5. GCN 的局限性

| 局限 | 说明 |
|------|------|
| 过平滑 | 层数增加后节点表示趋于相同 |
| 过拟合小图 | 在小图上容易过拟合 |
| 转导式 | 不能泛化到新节点（需要重新训练） |
| 同质性假设 | 假设相连节点相似，异构图上效果差 |
| 表达力有限 | 无法区分某些非同构图（如正则图） |

```python
# GCN 在 Cora 上的训练
from torch_geometric.datasets import Planetoid
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

model = GCNModel(dataset.num_node_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f'Epoch {epoch}: Loss={loss:.4f}, Test Acc={acc:.4f}')
```

## 6. 小结

- 谱图卷积理论基础：拉普拉斯特征分解定义图傅里叶变换
- ChebNet 用切比雪夫多项式避免特征分解
- GCN（K=1 进一步简化）成为最经典的 GNN 基线
- GCN 公式 $H^{(l+1)} = \sigma(\hat{D}^{-1/2} \hat{A} \hat{D}^{-1/2} H^{(l)} W^{(l)})$
- GCN 局限性：过平滑、转导式、同质性假设
