# 16_PointNet 与点云图网络

## 1. 点云数据

点云是由三维空间中的无序点集合构成的数据：

$$\mathcal{P} = \{p_1, p_2, \dots, p_N\}, \quad p_i \in \mathbb{R}^3 \text{ 或 } \mathbb{R}^{3+C}$$

其中 $C$ 是附加特征（颜色、法向量等）。

**点云的特点**：
- **无序性**：点的排列顺序不改变点云的语义
- **置换不变性**：$f(\{p_1, p_2\}) = f(\{p_2, p_1\})$
- **非结构化**：没有规则的网格结构
- **密度不均匀**：不同区域点的密度不同

**应用场景**：3D 物体识别、自动驾驶、室内导航、机器人抓取

## 2. PointNet

Qi et al. (2017)：直接处理无序点集的深度学习架构。

### 2.1 核心思想

用**对称函数**（如 max pooling）实现置换不变性：

$$f(\{x_1, \dots, x_N\}) = \text{MAX}\left(h(x_1), h(x_2), \dots, h(x_N)\right)$$

其中 $h$ 是逐点的 MLP。

### 2.2 架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNet(nn.Module):
    """PointNet 分类网络"""
    def __init__(self, num_classes=40, input_dim=3):
        super().__init__()
        # 逐点特征提取 (共享 MLP)
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        # 分类头
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, N, 3] -> [B, 3, N] (Conv1d 需要通道在前)
        x = x.transpose(2, 1)

        # 逐点 MLP
        x = F.relu(self.bn1(self.conv1(x)))    # [B, 64, N]
        x = F.relu(self.bn2(self.conv2(x)))     # [B, 128, N]
        x = self.bn3(self.conv3(x))             # [B, 1024, N]

        # 对称操作：全局最大池化（实现置换不变性）
        x = torch.max(x, dim=2)[0]             # [B, 1024]

        # 分类
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
```

### 2.3 T-Net（输入变换网络）

PointNet 使用一个小型网络预测一个变换矩阵，将输入对齐到规范空间：

$$T = \text{T-Net}(x) \in \mathbb{R}^{k \times k}$$

$$\tilde{x}_i = T \cdot x_i$$

```python
class TNet(nn.Module):
    """输入变换网络"""
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: [B, k, N]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, dim=2)[0]  # [B, 1024]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # [B, k*k]

        # 初始化为单位矩阵 + 学到的扰动
        identity = torch.eye(self.k, device=x.device).unsqueeze(0).repeat(x.size(0), 1, 1)
        x = x.view(-1, self.k, self.k) + identity
        return x
```

### 2.4 带 T-Net 的完整 PointNet

```python
class PointNetFull(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        self.input_tnet = TNet(k=3)
        self.feature_tnet = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # x: [B, N, 3]
        x = x.transpose(2, 1)  # [B, 3, N]

        # 输入变换
        T = self.input_tnet(x)  # [B, 3, 3]
        x = torch.bmm(T, x)     # [B, 3, N]

        # 逐点特征
        x = F.relu(self.conv1(x))  # [B, 64, N]
        x = F.relu(self.conv2(x))  # [B, 64, N]

        # 特征变换
        T_feat = self.feature_tnet(x)  # [B, 64, 64]
        x = torch.bmm(T_feat, x)       # [B, 64, N]

        # 更多逐点层
        x = F.relu(self.conv3(x))  # [B, 64, N]
        x = F.relu(self.conv4(x))  # [B, 128, N]
        x = self.conv5(x)          # [B, 1024, N]

        # 全局最大池化
        x = torch.max(x, dim=2)[0]  # [B, 1024]

        # 分类
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        return self.fc3(x)
```

## 3. PointNet++：层次化点云学习

Qi et al. (2017)：PointNet 缺乏局部结构感知。PointNet++ 引入**层次化**特征学习。

### 3.1 Set Abstraction 模块

三个步骤：
1. **采样（FPS）**：最远点采样选择中心点
2. **分组（Ball Query）**：以中心点为球心，采样邻居
3. **PointNet**：对每个局部区域应用 PointNet

```python
import torch
from torch_geometric.nn import fps, knn, knn_graph

class SetAbstraction(nn.Module):
    def __init__(self, in_dim, out_dim, ratio, radius):
        super().__init__()
        self.ratio = ratio
        self.radius = radius
        self.mlp = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, 1),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Conv1d(out_dim, out_dim, 1),
        )

    def forward(self, x, pos):
        # pos: [B, N, 3], x: [B, N, d]
        B, N, _ = pos.shape

        # 最远点采样（选择中心点）
        idx = fps(pos, batch=None, ratio=self.ratio)  # [B * num_centroids]
        centroids_pos = pos[0][idx]  # 简化版

        # Ball query：找每个中心点的邻居
        # 使用 kNN 近似
        row, col = knn(pos[0], centroids_pos, k=32)  # 32 个邻居

        # 局部 PointNet
        # 分组特征: [num_centroids, 32, d]
        grouped_x = x[0][col].view(len(idx), 32, -1)
        grouped_x = grouped_x.transpose(2, 1)  # [num_centroids, d, 32]
        local_feat = self.mlp(grouped_x)  # [num_centroids, out_dim, 32]
        local_feat = torch.max(local_feat, dim=2)[0]  # [num_centroids, out_dim]

        return local_feat, centroids_pos
```

## 4. 将点云构建为图

点云可以自然地转化为 $k$-NN 图，然后使用图神经网络处理。

```python
from torch_geometric.nn import knn_graph, GCNConv
from torch_geometric.data import Data

def pointcloud_to_graph(points, k=16):
    """将点云转为 k-NN 图"""
    edge_index = knn_graph(points, k=k, loop=False)
    return Data(x=points, pos=points, edge_index=edge_index)

class PointGNN(nn.Module):
    """用 GNN 处理点云"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        # 全局池化
        x = global_max_pool(x, data.batch)
        return self.lin(x)
```

## 5. 点云分类任务

```python
from torch_geometric.datasets import ModelNet
from torch_geometric.transforms import SamplePoints
from torch_geometric.loader import DataLoader

# ModelNet40 数据集：40 类 3D 物体
dataset = ModelNet(root='/tmp/ModelNet', name='40', transform=SamplePoints(1024))
train_loader = DataLoader(dataset[:int(len(dataset)*0.8)], batch_size=32, shuffle=True)
test_loader = DataLoader(dataset[int(len(dataset)*0.8):], batch_size=32)

model = PointNet(num_classes=40)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

for epoch in range(50):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.pos)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    print(f'Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}')
```

## 6. PointNet vs 图网络

| 特性 | PointNet | GNN (on k-NN graph) |
|------|----------|---------------------|
| 局部结构 | 无 | 有（k-NN 邻域） |
| 置换不变性 | 全局 max pool | 聚合函数 |
| 密度适应 | 差 | 较好 |
| 计算效率 | 高 | 中等 |
| 表达力 | 受限于全局池化 | 更强 |

## 7. 小结

- 点云是无序的 3D 点集合，核心挑战是置换不变性
- PointNet 通过逐点 MLP + 全局 max pool 实现置换不变性
- T-Net 学习输入/特征的对齐变换
- PointNet++ 引入层次化学习：FPS 采样 + Ball Query + 局部 PointNet
- 点云可转化为 k-NN 图，用 GNN 处理以捕获局部结构
