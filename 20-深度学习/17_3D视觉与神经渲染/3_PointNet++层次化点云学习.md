# 3_PointNet++ 层次化点云学习

## 1. PointNet的局限与PointNet++的动机

PointNet对每个点独立处理后做全局池化，缺乏**局部几何结构**的建模能力。类比CNN的启发：通过层层堆叠，从局部到全局提取特征。

**PointNet++ (Qi et al., NeurIPS 2017)** 将层次化学习引入点云处理，核心思想来自CNN：**局部特征提取 → 池化 → 更大范围特征提取**。

## 2. Set Abstraction 模块

### 2.1 三层流程

每个 Set Abstraction (SA) 层包含三步：

1. **采样 (Sampling)**：用最远点采样 (FPS) 选择 $N'$ 个中心点
2. **分组 (Grouping)**：以每个中心点为球心，半径 $r$ 内的点组成局部邻域
3. **PointNet 特征提取**：对每个邻域用 mini-PointNet 提取特征

```
输入: (N, 3+C)  N个点，3坐标+C维特征
    ↓ FPS采样
中心点: (N', 3+C)  选择N'个代表性点
    ↓ 球查询分组
邻域: (N', K, 3+C)  每个中心点的K个邻居
    ↓ mini-PointNet
输出: (N', 3+C')  新的点集，更少但更丰富
```

### 2.2 最远点采样 (Farthest Point Sampling)

FPS 保证采样点均匀覆盖整个点云：

```python
def farthest_point_sampling(points, n_samples):
    """
    最远点采样
    points: (B, N, 3)
    返回: (B, n_samples) 采样点索引
    """
    B, N, _ = points.shape
    centroids = torch.zeros(B, n_samples, dtype=torch.long)
    distance = torch.full((B, N), float('inf'))
    
    # 随机选择第一个点
    farthest = torch.randint(0, N, (B,))
    
    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = points[torch.arange(B), farthest].unsqueeze(1)  # (B, 1, 3)
        # 计算所有点到当前采样点的距离
        dist = torch.sum((points - centroid) ** 2, dim=-1)  # (B, N)
        # 更新最小距离
        distance = torch.min(distance, dist)
        # 选择距离最大的点作为下一个采样点
        farthest = torch.argmax(distance, dim=-1)
    
    return centroids
```

**时间复杂度**：$O(N \cdot N')$，优于随机采样的覆盖效果。

### 2.3 球查询 (Ball Query)

```python
def ball_query(points, center, radius, k):
    """
    球查询分组
    points: (B, N, 3+C) 所有点
    center: (B, N', 3) 中心点坐标
    返回: (B, N', K, 3+C)
    """
    B, N, _ = points.shape
    N_ = center.shape[1]
    
    # 计算每个中心点到所有点的距离
    dist = torch.cdist(center[:, :, :3], points[:, :, :3])  # (B, N', N)
    
    # 球查询：距离小于radius的点
    group_mask = dist < radius  # (B, N', N)
    
    # 对每个中心点，取最近的K个点（在球内）
    dist[~group_mask] = float('inf')
    _, knn_idx = dist.topk(k, largest=False, dim=-1)  # (B, N', K)
    
    # 收集特征
    knn_idx_exp = knn_idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1])
    grouped = torch.gather(points.unsqueeze(1).expand(-1, N_, -1, -1), 2, knn_idx_exp)
    
    # 相对坐标（归一化）
    grouped[:, :, :, :3] -= center[:, :, :3].unsqueeze(2)
    
    return grouped  # (B, N', K, 3+C)
```

### 2.4 完整 SA 模块

```python
class SetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        """
        npoint: 采样点数
        radius: 球查询半径
        nsample: 每个球内的采样数
        mlp: MLP通道列表 [64, 128, 256]
        """
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        
        layers = []
        last_channel = in_channel + 3  # 相对坐标拼接特征
        for out_ch in mlp:
            layers.extend([
                nn.Conv2d(last_channel, out_ch, 1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            ])
            last_channel = out_ch
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, xyz, features):
        # xyz: (B, N, 3), features: (B, N, C)
        # 1. FPS采样
        idx = farthest_point_sampling(xyz, self.npoint)
        new_xyz = xyz[torch.arange(xyz.shape[0]).unsqueeze(1), idx]  # (B, npoint, 3)
        
        # 2. 球查询分组
        grouped = ball_query(
            torch.cat([xyz, features], dim=-1) if features is not None else xyz,
            new_xyz, self.radius, self.nsample
        )  # (B, npoint, nsample, 3+C)
        
        # 3. 逐邻域PointNet
        grouped = grouped.permute(0, 3, 1, 2)  # (B, 3+C, npoint, nsample)
        new_features = self.mlp(grouped)  # (B, C', npoint, nsample)
        new_features = torch.max(new_features, dim=-1)[0]  # (B, C', npoint)
        
        return new_xyz, new_features.permute(0, 2, 1)  # (B, npoint, 3), (B, npoint, C')
```

## 3. 多尺度分组 (MSG) 与多分辨率分组 (MRG)

### 3.1 问题：非均匀密度

真实点云密度不均匀（近密远疏），单一尺度的球查询在稀疏区域可能捕获不到足够邻域。

### 3.2 MSG (Multi-Scale Grouping)

每个中心点使用**多个半径**查询，拼接特征：

$$\mathbf{f}_i = \text{Concat}\left[\text{PointNet}(G(x_i, r_1)), \text{PointNet}(G(x_i, r_2)), \ldots\right]$$

### 3.3 MRG (Multi-Resolution Grouping)

结合当前层的局部特征和前一层的全局特征：

```
当前层局部特征: PointNet(当前层邻域)
       +
上一层全局特征: 上一层SA的输出
       ↓
    拼接
```

MRG 比 MSG 更高效，因为复用了已有计算。

## 4. 特征传播 (Feature Propagation)

用于分割任务的上采样模块，将特征传播回原始分辨率：

```python
class FeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_ch in mlp:
            layers.extend([
                nn.Conv1d(last_channel, out_ch, 1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU()
            ])
            last_channel = out_ch
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, xyz1, xyz2, feat1, feat2):
        """
        xyz1: (B, N, 3) 较密的点
        xyz2: (B, M, 3) 较疏的点（SA输出）
        feat1: (B, N, C1) 对应xyz1的特征
        feat2: (B, M, C2) 对应xyz2的特征
        """
        # 1. 三线性插值上采样
        B, N, _ = xyz1.shape
        dist = torch.cdist(xyz1, xyz2)  # (B, N, M)
        dist, idx = dist.sort(dim=-1)
        dist, idx = dist[:, :, :3], idx[:, :, :3]  # 最近3个点
        
        # 距离倒数加权插值
        weight = 1.0 / (dist + 1e-8)
        weight = weight / weight.sum(dim=-1, keepdim=True)
        
        interpolated = torch.gather(
            feat2.unsqueeze(1).expand(-1, N, -1, -1), 2,
            idx.unsqueeze(-1).expand(-1, -1, -1, feat2.shape[-1])
        )  # (B, N, 3, C2)
        interpolated = (interpolated * weight.unsqueeze(-1)).sum(dim=2)  # (B, N, C2)
        
        # 2. 跳跃连接拼接
        if feat1 is not None:
            new_feat = torch.cat([feat1, interpolated], dim=-1)
        else:
            new_feat = interpolated
        
        # 3. 逐点MLP
        new_feat = self.mlp(new_feat.transpose(2, 1)).transpose(2, 1)
        return new_feat
```

## 5. 完整 PointNet++ 网络

### 5.1 分类网络

```python
class PointNetPlusPlus(nn.Module):
    def __init__(self, num_classes=40):
        super().__init__()
        # SA层（逐步降采样）
        self.sa1 = SetAbstraction(512, 0.2, 32, 3 + 0, [64, 64, 128])
        self.sa2 = SetAbstraction(128, 0.4, 64, 128 + 3, [128, 128, 256])
        self.sa3 = SetAbstraction(None, None, None, 256 + 3, [256, 512, 1024])
        
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, xyz):
        B, N, _ = xyz.shape
        l0_xyz = xyz
        l0_feat = None
        
        l1_xyz, l1_feat = self.sa1(l0_xyz, l0_feat)
        l2_xyz, l2_feat = self.sa2(l1_xyz, l1_feat)
        l3_xyz, l3_feat = self.sa3(l2_xyz, l2_feat)
        
        global_feat = l3_feat.squeeze(1)  # (B, 1024)
        return self.classifier(global_feat)
```

### 5.2 分割网络 (SSG)

```
输入: (B, N, 3)
  ↓ SA1: N → 512
  ↓ SA2: 512 → 128
  ↓ SA3: 128 → 1
  ↓ FP3: 1 → 128 + skip
  ↓ FP2: 128 → 512 + skip
  ↓ FP1: 512 → N + skip
输出: (B, N, num_classes)
```

## 6. MSG vs SSG 对比

| 配置 | 缩写 | 特点 | 适用场景 |
|------|------|------|----------|
| Single Scale Grouping | SSG | 每层单一半径 | 均匀密度数据 |
| Multi-Scale Grouping | MSG | 每层多半径拼接 | 非均匀密度 |
| Multi-Resolution Grouping | MRG | 复用上层特征 | 大规模点云 |

## 7. 性能对比

| 方法 | ModelNet40 分类 (%) | ScanNet 分割 (mIoU) |
|------|---------------------|---------------------|
| PointNet | 89.2 | 73.9 |
| PointNet++ (SSG) | 90.7 | — |
| PointNet++ (MSG) | 91.9 | 84.3 |

---

**关键要点**：
1. PointNet++ 通过采样-分组-特征提取的层次化设计，解决了PointNet缺乏局部结构感知的问题
2. FPS + 球查询是点云下采样的标准组合
3. MSG/MRG 应对非均匀点密度的挑战
4. 特征传播（插值+拼接+MLP）实现了从稀疏到稠密的特征上采样
