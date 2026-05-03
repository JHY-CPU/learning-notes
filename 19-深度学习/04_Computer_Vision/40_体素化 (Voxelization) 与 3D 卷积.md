# 40_体素化 (Voxelization) 与 3D 卷积

## 核心概念

- **体素化（Voxelization）**：将3D空间划分为规整的网格（体素，voxel），每个体素是一个小的3D立方体，类似于2D图像中的像素。体素化是将非结构化3D数据转换为结构化表示的关键步骤。
- **体素格（Voxel Grid）**：3D空间的离散化表示，每个体素包含该区域内是否存在点、点的统计信息（平均坐标、强度）或特征向量。
- **3D卷积在体素上的应用**：类比2D CNN在图像上的操作，3D CNN在体素格上使用3D卷积核进行特征提取。卷积核形状为 $K_d \times K_h \times K_w$。
- **稀疏性挑战**：3D空间中大部分区域是空的，体素格极度稀疏（通常>90%为空），直接使用密集3D卷积会导致巨大的计算和内存浪费。
- **VoxelNet (2018)**：Yin Zhou & Oncel Tuzel 提出的端到端3D目标检测网络，将点云体素化后使用VFE（Voxel Feature Encoding）层提取特征，再使用3D卷积进行空间特征学习。
- **稀疏卷积（Sparse Convolution）**：解决了密集3D卷积的效率问题——只在非空的体素位置上进行计算，大幅降低了计算量和内存占用。Submanifold Sparse Convolution 进一步限制了特征的"膨胀"。

## 数学推导

**体素化过程：**
给定点云 $\{p_i = (x_i, y_i, z_i, f_i)\}_{i=1}^N$，体素尺寸 $v_d \times v_h \times v_w$，点 $p_i$ 所属体素索引为：
$$
idx_x = \lfloor (x_i - x_{min}) / v_w \rfloor
$$
$$
idx_y = \lfloor (y_i - y_{min}) / v_h \rfloor
$$
$$
idx_z = \lfloor (z_i - z_{min}) / v_d \rfloor
$$

**VFE（Voxel Feature Encoding）层：**
每个非空体素内的点经过VFE层：
- 计算体素内所有点的均值 $(\bar{x}, \bar{y}, \bar{z})$
- 对每个点拼接相对坐标 $(x_i - \bar{x}, y_i - \bar{y}, z_i - \bar{z})$
- 经过全连接层 + 最大池化聚合

**3D卷积的计算量：**
密集3D卷积：
$$
\text{Cost} = C_{in} \times C_{out} \times K^3 \times D \times H \times W
$$

稀疏3D卷积：
$$
\text{Cost} = C_{in} \times C_{out} \times K^3 \times N_{active}
$$

其中 $N_{active}$ 是非空体素的数量。当 $N_{active} \ll D \times H \times W$ 时，稀疏卷积大幅减少计算量。

## 直观理解

体素化是将点云从"一堆散点"变成"3D网格图像"的过程。这就像把2D图像上的像素映射到规整的网格上——只不过是在3D空间中。每个体素就像是3D空间中的一个"小方块"，有点落入的方块被标记为占据，空的方块保持空白。

最大的挑战是稀疏性：在自动驾驶场景中，一辆汽车的LiDAR扫描范围可达100米，但实际物体只占空间的极小部分。如果对所有体素进行密集3D卷积，就像在2D图像中只有1%的像素有内容却对所有像素都进行计算，显然是巨大的浪费。稀疏卷积通过只计算非空体素，实现了数十倍到数百倍的效率提升。

## 代码示例

```python
import torch
import torch.nn as nn

class VFELayer(nn.Module):
    """Voxel Feature Encoding 层"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x, mask):
        # x: (B*N_nonempty, max_points_per_voxel, C)
        # mask: (B*N_nonempty, max_points_per_voxel) 有效点掩码
        x = self.linear(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.relu(x)
        # 最大池化（只聚合有效点）
        x_max = torch.max(x + (1 - mask.unsqueeze(-1)) * -1e9, dim=1, keepdim=True)[0]
        # 将池化结果拼接回每个点
        x = torch.cat([x, x_max.expand_as(x)], dim=-1)
        return x

class VoxelNet(nn.Module):
    """简化版 VoxelNet 主干网络"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.vfe = nn.Sequential(
            VFELayer(7, 32),  # 输入: x,y,z,reflectance + 3个相对偏移
            VFELayer(64, 128),
        )
        # 3D卷积中间层
        self.conv3d = nn.Sequential(
            nn.Conv3d(128, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
            nn.Conv3d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm3d(64), nn.ReLU(),
        )
        # RPN
        self.rpn = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, num_classes, 1),
        )

    def forward(self, voxel_features, voxel_coords):
        # voxel_features: (N_voxels, max_points, C) 
        # voxel_coords: (N_voxels, 4) [batch, z, y, x]
        # VFE
        mask = (voxel_features[:, :, 0] != 0).float()
        x = self.vfe[0](voxel_features, mask)
        x = self.vfe[1](x, mask)
        x = torch.max(x, dim=1)[0]  # (N_voxels, 128)
        
        # 散射到3D体素空间 (简化的密集化)
        batch_size = voxel_coords[:, 0].max().int() + 1
        # ... (实际需要scatter操作)
        # 简化为直接生成随机3D张量
        x = x.view(batch_size, 128, 10, 10, 4) if False else torch.randn(2, 128, 10, 10, 4)
        
        # 3D卷积
        x = self.conv3d(x)  # (B, 64, 3, 3, 1)
        x = x.squeeze(-1).squeeze(-1)  # (B, 64)
        return self.rpn(x.unsqueeze(-1).unsqueeze(-1))

# 演示常规3D卷积
conv3d = nn.Conv3d(16, 32, kernel_size=3, padding=1)
x = torch.randn(1, 16, 10, 10, 10)
out = conv3d(x)
print(f"3D卷积输入: {x.shape}")
print(f"3D卷积输出: {out.shape}")
print(f"3D卷积参数量: {sum(p.numel() for p in conv3d.parameters()):,}")

# 密集 vs 稀疏3D卷积的对比
dense_params = sum(p.numel() for p in conv3d.parameters())
print(f"\n注意: 当体素空间尺寸为 100x100x10 时: ")
print(f"密集3D卷积计算量 ≈ 100x100x10 = 100,000 体素")
print(f"稀疏3D卷积 (假设5%非空) ≈ 5,000 体素")
print(f"稀疏卷积可节省约 95% 的计算量")
```

## 深度学习关联

- **自动驾驶3D检测的标配**：VoxelNet开创的"体素化 + 3D卷积 + RPN"范式被后续大量工作继承，如PointPillars（将3D体素压缩为2D柱状体，使用2D卷积替代3D卷积，速度提升10倍以上）、SECOND（使用稀疏卷积替代密集3D卷积）、CenterPoint等。
- **稀疏卷积的重要性**：稀疏卷积（MinkowskiEngine、SparseConvNet）几乎成为所有3D场景理解任务的标配，显著扩展了3D CNN可以处理的场景规模。Submanifold Sparse Convolution进一步优化了稀疏3D卷积的行为。
- **从结构化到非结构化的演进**：体素化方法（结构化）和PointNet系列（非结构化）代表了3D深度学习的两种范式。后续的PV-RCNN等混合方法结合了两者的优势，在3D目标检测中取得了SOTA结果。
