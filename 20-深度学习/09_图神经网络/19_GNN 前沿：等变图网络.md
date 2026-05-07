# 19_GNN 前沿：等变图网络

## 1. 等变性的动机

在物理系统、分子模拟等领域，系统的**对称性**是重要的归纳偏置：
- 旋转一个分子，它的能量不变
- 平移一个分子，力的大小和方向跟着平移

**等变性（Equivariance）**：输入变换后，输出以相应方式变换。

$$f(T(x)) = T'(f(x))$$

对于分子系统：
- **不变量**：能量 $E$（旋转分子，能量不变）
- **等变量**：力 $\vec{F} = -\nabla E$（旋转分子，力向量跟着旋转）

## 2. 数学基础

### 2.1 群与群作用

**群（Group）** $G$：满足封闭性、结合律、单位元、逆元的代数结构。

**群作用**：群 $G$ 作用在空间 $X$ 上，记为 $g \cdot x$。

常见群：
- 平移群 $\mathbb{R}^n$
- 旋转群 $SO(2)$, $SO(3)$
- 欧几里得群 $E(n) = SO(n) \ltimes \mathbb{R}^n$（旋转 + 平移）
- 特殊欧几里得群 $SE(3) = SO(3) \ltimes \mathbb{R}^3$

### 2.2 等变性的形式定义

函数 $f: X \rightarrow Y$ 是 $G$-等变的，如果：

$$f(g \cdot x) = \rho_Y(g) \cdot f(x), \quad \forall g \in G$$

其中 $\rho_X, \rho_Y$ 分别是 $G$ 在 $X$ 和 $Y$ 上的表示。

**特殊情况——不变性**：

$$f(g \cdot x) = f(x)$$

即 $\rho_Y$ 是平凡表示。

## 3. E(n)-等变图网络

Satorras et al. (2021)：构建对欧几里得变换等变的 GNN。

### 3.1 节点表示更新

$$h_i^{(l+1)} = \phi_h\left(h_i^{(l)}, \bigoplus_{j \in \mathcal{N}(i)} \phi_e(h_i^{(l)}, h_j^{(l)}, \|x_i^{(l)} - x_j^{(l)}\|^2)\right)$$

### 3.2 坐标更新（关键）

坐标更新必须是等变的：

$$x_i^{(l+1)} = x_i^{(l)} + \sum_{j \neq i} (x_i^{(l)} - x_j^{(l)}) \phi_x(h_i^{(l)}, h_j^{(l)}, \|x_i^{(l)} - x_j^{(l)}\|^2)$$

其中 $\phi_x$ 是标量函数，$(x_i - x_j)$ 是位移向量。标量函数在旋转下不变，位移向量在旋转下等变，所以整个更新是等变的。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class EGNNLayer(nn.Module):
    """E(n)-等变图神经网络层"""
    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        # 边网络：输入两节点特征 + 距离，输出边特征
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim),
        )

        # 节点更新
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim + edge_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 坐标更新（标量权重）
        self.coord_mlp = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )

    def forward(self, h, x, edge_index):
        """
        h: [N, d] 节点特征
        x: [N, 3] 坐标
        edge_index: [2, E]
        """
        src, dst = edge_index

        # 计算位移和距离
        diff = x[src] - x[dst]       # [E, 3]
        dist = (diff ** 2).sum(-1, keepdim=True)  # [E, 1]

        # 边特征
        edge_feat = self.edge_mlp(torch.cat([h[src], h[dst], dist], dim=-1))  # [E, d_e]

        # 节点更新（消息传递）
        aggr = torch.zeros(h.size(0), edge_feat.size(1), device=h.device)
        aggr.scatter_add_(0, dst.unsqueeze(-1).expand_as(edge_feat), edge_feat)
        h = h + self.node_mlp(torch.cat([h, aggr], dim=-1))

        # 坐标更新（等变）
        coord_weight = self.coord_mlp(edge_feat)  # [E, 1]
        coord_update = coord_weight * diff  # [E, 3] — 等变：标量 × 向量

        x_update = torch.zeros_like(x)
        x_update.scatter_add_(0, dst.unsqueeze(-1).expand_as(coord_update), coord_update)
        x = x + x_update

        return h, x


class EGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4):
        super().__init__()
        self.h_embed = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            EGNNLayer(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, h, x, edge_index):
        h = self.h_embed(h)
        for layer in self.layers:
            h, x = layer(h, x, edge_index)
        return self.output(h), x  # h: 不变特征, x: 等变坐标
```

## 4. SE(3)-等变网络

SE(3) 是 3D 空间中的旋转 + 平移群。SE(3)-等变网络专门用于分子和蛋白质建模。

### 4.1 表示类型

在 SE(3) 等变网络中，特征分为不同阶：
- **标量（0 阶）**：$l = 0$，旋转变换下不变
- **向量（1 阶）**：$l = 1$，旋转变换下等变
- **高阶张量**：$l \geq 2$

### 4.2 TFN（Tensor Field Networks）

Thomas et al. (2018)：基于球谐函数的 SE(3)-等变卷积。

$$[h_i * g]_m^{(l)} = \sum_{j \in \mathcal{N}(i)} \sum_{l_1, l_2} W^{(l_1, l_2, l)}(\|r_{ij}\|) \sum_{m_1, m_2} C_{l_1 m_1, l_2 m_2}^{l m} Y_{l_2 m_2}(\hat{r}_{ij}) h_{j, m_1}^{(l_1)}$$

其中 $C_{l_1 m_1, l_2 m_2}^{l m}$ 是 Clebsch-Gordan 系数。

## 5. 分子动力学应用

等变网络在分子动力学中至关重要：预测原子受力，进行分子模拟。

### 5.1 任务形式化

$$E = f(\{z_i, x_i\}_{i=1}^N), \quad F_i = -\frac{\partial E}{\partial x_i}$$

- $z_i$：原子类型（不变特征）
- $x_i$：原子坐标（等变特征）
- $E$：系统能量（不变标量）
- $F_i$：原子受力（等变向量）

```python
class MolecularDynamics(nn.Module):
    """用 EGNN 做分子动力学"""
    def __init__(self, num_atom_types, hidden_dim, num_layers=6):
        super().__init__()
        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
        self.egnn = EGNN(hidden_dim, hidden_dim, hidden_dim, num_layers)
        self.energy_head = nn.Linear(hidden_dim, 1)

    def forward(self, atom_types, positions, edge_index):
        # 原子嵌入
        h = self.atom_embed(atom_types)

        # EGNN 前向
        h_out, pos_out = self.egnn(h, positions, edge_index)

        # 全局能量（不变标量）
        energy = self.energy_head(h_out).sum(dim=0)  # 标量

        # 力：能量对坐标的梯度（自动等变）
        positions.requires_grad_(True)
        energy_scalar = self.energy_head(h_out).sum()
        forces = -torch.autograd.grad(energy_scalar, positions, create_graph=True)[0]

        return energy, forces

# 训练循环
model = MolecularDynamics(num_atom_types=10, hidden_dim=128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    energy_pred, forces_pred = model(atom_types, positions, edge_index)

    # 损失：能量和力的回归
    energy_loss = F.mse_loss(energy_pred, energy_target)
    forces_loss = F.mse_loss(forces_pred, forces_target)
    loss = energy_loss + 100 * forces_loss  # 力的权重更大

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 6. 等变网络的实用场景

| 场景 | 等变性要求 | 典型方法 |
|------|-----------|----------|
| 分子能量预测 | SO(3)/SE(3) 不变 | EGNN, SchNet |
| 分子力预测 | SO(3)/SE(3) 等变 | EGNN, PaiNN |
| 蛋白质结构预测 | SE(3) 等变 | AlphaFold, ESMFold |
| 点云分类 | 排列不变 | PointNet |
| 3D 物体检测 | SO(3) 等变 | SE(3)-Transformers |
| 流体力学 | E(3) 等变 | EGNN |

## 7. PyG 中的等变网络

```python
from torch_geometric.nn import EGNNConv

class PyGEGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4):
        super().__init__()
        self.node_enc = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNConv(hidden_dim, hidden_dim, hidden_dim))

        self.output = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, pos, edge_index):
        h = self.node_enc(x)
        for conv in self.convs:
            h, pos = conv(h, pos, edge_index)
        return self.output(h)
```

## 8. 前沿进展

| 模型 | 年份 | 特点 |
|------|------|------|
| SchNet | 2017 | 连续距离滤波器，不变 |
| DimeNet | 2020 | 使用角度信息，不变 |
| EGNN | 2021 | 简洁的等变坐标更新 |
| PaiNN | 2021 | 向量通道，高效等变 |
| Equiformer | 2022 | 等变注意力 |
| MACE | 2022 | 高阶等变消息传递 |

## 9. 小结

- 等变性是对称性的数学表达：输入变换后输出相应变换
- E(n)-等变网络通过标量函数更新坐标偏移，天然保证等变性
- SE(3)-等变网络使用球谐函数和 Clebsch-Gordan 系数
- 分子动力学是等变网络的核心应用：预测能量（不变）和力（等变）
- 等变性作为归纳偏置，大幅提升样本效率和泛化能力
