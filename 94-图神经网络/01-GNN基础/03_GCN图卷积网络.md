# GCN图卷积网络


## 1. 从 CNN 到 GCN 的动机


卷积神经网络（CNN）在图像、文本等规则数据上取得了巨大成功，但这些数据可以看作特殊的图——像素网格或序列。对于一般的图结构数据，需要将卷积推广到不规则的图上。


### CNN vs 图数据的挑战


| 特性 | 规则网格（图像） | 图结构数据 |
| --- | --- | --- |
| 邻居数量 | 固定（如3×3卷积核） | 不固定，每个节点不同 |
| 节点顺序 | 天然有序（上/下/左/右） | 无序 |
| 空间距离 | 均匀 | 不均匀 |
| 全局结构 | 规则网格 | 任意拓扑 |


### 核心问题


如何在图上定义"卷积"操作？两种思路：


- **谱方法（Spectral）**
   ：基于图傅里叶变换，在频域定义卷积
- **空间方法（Spatial）**
   ：直接在空间域聚合邻居信息


## 2. 谱方法 vs 空间方法


### 谱方法


基于图信号处理理论，利用拉普拉斯矩阵的特征分解进行卷积。


$$
图傅里叶变换: x̂ = UT x
                谱卷积: g * x = U gθ(Λ) UT x
                其中 U 是拉普拉斯矩阵 L 的特征向量矩阵，Λ 是特征值对角矩阵
$$


**缺点：**需要特征分解 O(n³)，计算量大；滤波器与图结构耦合，不能迁移到其他图。


### 空间方法


直接在图的空间域上聚合邻居节点的信息，更直观高效。


- 直观：类似于CNN在像素邻域上操作
- 高效：不需要矩阵分解
- 可迁移：学到的滤波器可在不同图上使用


### 谱方法到空间方法的桥梁


Kipf & Welling (2017) 通过对谱卷积的近似，得到了一阶近似GCN，本质上等价于空间聚合方法。


| 方法 | 代表模型 | 计算复杂度 | 可迁移性 |
| --- | --- | --- | --- |
| 谱方法 | Spectral CNN, ChebNet | 高（需特征分解） | 差 |
| 空间方法 | GCN, GAT, GraphSAGE | 低 | 好 |


## 3. GCN 核心公式


### 前向传播公式


$$
H(l+1) = σ(D̃-1/2 Ã D̃-1/2 H(l) W(l))
$$


其中：


- **Ã = A + I**
   ：添加自环后的邻接矩阵（让每个节点也聚合自身特征）
- **D̃**
   ：Ã 的度矩阵，D̃
   ~ii~
   = Σ
   ~j~
   Ã
   ~ij~
- **D̃^-1/2^ Ã D̃^-1/2^**
   ：对称归一化邻接矩阵
- **H^(l)^**
   ：第 l 层的节点特征矩阵，H
   ^(0)^
   = X
- **W^(l)^**
   ：第 l 层的可学习权重矩阵
- **σ**
   ：激活函数（如 ReLU）


### 归一化技巧详解


> **Example:** #### 为什么需要自环？
>
>
> 原始邻接矩阵 A 的对角线为 0，节点不会聚合自身特征。添加自环 Ã = A + I 后，每个节点在聚合时包含自身。
>
>
> #### 为什么用对称归一化？
>
>
> 简单均值归一化 D^-1^A 会导致高度节点对邻居的影响过大。对称归一化 D^-1/2^AD^-1/2^ 使得：
>                 归一化后 A̅~ij~ = A~ij~ / √(d~i~ · d~j~)，对高低度节点都公平。


### 单节点视角


$$
hv(l+1) = σ(Σu∈N(v)∪{v} (1/√(d̃v · d̃u)) · hu(l) · W(l))
$$


## 4. GCN 的数学推导（Kipf & Welling 2017）


### 从谱卷积出发


1. **谱卷积：**
   g
   ~θ~
   * x = U g
   ~θ~
   (Λ) U
   ^T^
   x
2. **ChebNet近似：**
   用切比雪夫多项式 K 阶展开近似谱滤波器
3. **一阶近似：**
   令 K=1，并对特征值范围做约束


### ChebNet 到 GCN 的简化


$$
ChebNet: gθ * x ≈ Σk=0K θk Tk(L̃) x
                K=1时: gθ * x ≈ θ0 x + θ1 (L - I) x = θ0 x - θ1 D-1/2 A D-1/2 x
                进一步约束 θ0 = -θ1 得到：
                gθ * x ≈ θ (I + D-1/2 A D-1/2) x
$$


> **Note:** **重归一化技巧：**
> I + D
> ^-1/2^
> A D
> ^-1/2^
> → D̃
> ^-1/2^
> Ã D̃
> ^-1/2^
> ，其中 Ã = A + I，D̃ 是 Ã 的度矩阵。这个替换在实践中能显著改善梯度传播。


## 5. GCN 的局限性


### 无法区分一阶邻居相同的节点


GCN 的聚合方式基于度归一化的均值，如果两个节点的一阶邻居完全相同（或结构等价），GCN 无法区分它们。


> **Example:** #### 例子：两个度数相同但邻居不同的节点
>
>
> 考虑一个规则图（如正方形网格），每个内部节点有4个邻居。如果所有节点的初始特征相同，经过GCN后所有节点的表示仍然相同。


### 其他局限


| 局限性 | 描述 | 解决方案 |
| --- | --- | --- |
| 固定权重 | 所有邻居权重相同，无法区分重要性 | GAT（注意力权重） |
| 过平滑 | 多层叠加后节点表示趋同 | 残差连接、DropEdge |
| 转导式 | 需要完整的邻接矩阵，不能处理新节点 | GraphSAGE（采样聚合） |
| 同配性假设 | 假设相连节点标签相同 | GPR-GNN, H2GCN |


## 6. PyTorch 代码实现


### 手写 GCN 层


```
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    """手写GCN层"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X, A):
        """
        X: [N, in_features] 节点特征
        A: [N, N] 归一化邻接矩阵（已添加自环并归一化）
        """
        # 线性变换
        X = self.linear(X)
        # 邻居聚合（矩阵乘法等价于消息传递）
        X = torch.mm(A, X)
        return X


def normalize_adjacency(A):
    """计算对称归一化邻接矩阵 D̃^{-1/2} Ã D̃^{-1/2}"""
    # 添加自环
    A_hat = A + torch.eye(A.size(0))
    # 度矩阵
    D_hat = torch.diag(A_hat.sum(dim=1))
    # D^{-1/2}
    D_inv_sqrt = torch.diag(torch.pow(D_hat.diag(), -0.5))
    D_inv_sqrt[D_inv_sqrt == float('inf')] = 0
    # 对称归一化
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
    return A_norm


# ========== 使用示例 ==========
# 创建一个简单图
A = torch.tensor([
    [0,1,1,0,0],
    [1,0,1,0,0],
    [1,1,0,1,0],
    [0,0,1,0,1],
    [0,0,0,1,0]
], dtype=torch.float)

X = torch.randn(5, 16)  # 5个节点，16维特征
A_norm = normalize_adjacency(A)

# 单层GCN
gcn_layer = GCNLayer(16, 32)
out = gcn_layer(X, A_norm)
print(f"GCN输出形状: {out.shape}")  # [5, 32]
```


### PyTorch Geometric 使用 GCN


```
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid

# 加载Cora数据集
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(dataset.num_features, 16, dataset.num_classes).to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        model.eval()
        pred = model(data).argmax(dim=1)
        correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        acc = int(correct) / int(data.test_mask.sum())
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Test Acc: {acc:.4f}')
        model.train()
```


## 总结


- GCN通过一阶谱卷积近似，等价于对邻居特征的归一化均值聚合
- 核心公式：H
   ^(l+1)^
   = σ(D̃
   ^-1/2^
   ÃD̃
   ^-1/2^
   H
   ^(l)^
   W
   ^(l)^
   )
- 自环 Ã=A+I 让节点聚合自身特征；对称归一化防止高度节点主导
- GCN的局限：固定权重、过平滑、转导式学习
- GCN是GNN领域最重要的基线模型之一，理解其原理是学习后续GNN的基础


<!-- Converted from: 03_GCN图卷积网络.html -->
