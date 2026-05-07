# 9_NeRF 变体与加速

## 1. NeRF 的局限性

原始 NeRF 存在以下主要问题：

| 问题 | 具体表现 | 影响 |
|------|----------|------|
| 训练慢 | 单场景需1-2天 | 不实用 |
| 渲染慢 | 每帧需数十秒 | 无法实时 |
| 场景特定 | 每个场景单独训练 | 不能泛化 |
| 静态场景 | 不支持动态 | 应用受限 |

本章介绍解决这些问题的主要变体。

## 2. Instant-NGP (Instant Neural Graphics Primitives)

### 2.1 核心思想

**Instant-NGP (Muller et al., SIGGRAPH 2022)** 使用**多分辨率哈希编码**替代大型MLP，将训练时间从天降到秒。

### 2.2 多分辨率哈希编码

传统 NeRF 用位置编码 + MLP。Instant-NGP 的思路：

1. 将空间在多个分辨率下划分为网格
2. 每个网格顶点存储一个可学习的特征向量
3. 使用哈希表高效存储（不存储空区域）
4. 插值得到查询点的特征

$$\gamma_\theta(\mathbf{x}) = \text{HashLookup}(\mathbf{x}, \text{level}=l) \quad \text{for } l = 1, \ldots, L$$

```python
import torch
import torch.nn as nn

class HashEncoder(nn.Module):
    """多分辨率哈希编码"""
    def __init__(self, num_levels=16, base_res=16, finest_res=512, 
                 hash_size=2**19, feat_dim=2):
        super().__init__()
        self.num_levels = num_levels
        self.feat_dim = feat_dim
        self.hash_size = hash_size
        
        # 每层分辨率（对数增长）
        self.b = (finest_res / base_res) ** (1 / (num_levels - 1))
        self.resolutions = [int(base_res * (self.b ** i)) for i in range(num_levels)]
        
        # 哈希表：每层一个
        self.hash_tables = nn.ParameterList([
            nn.Parameter(torch.randn(hash_size, feat_dim) * 0.0001)
            for _ in range(num_levels)
        ])
    
    def spatial_hash(self, coords):
        """空间哈希函数"""
        # xor哈希
        primes = torch.tensor([1, 2654435761, 805459861], device=coords.device)
        h = (coords * primes).sum(dim=-1)
        return h % self.hash_size
    
    def forward(self, x):
        # x: (N, 3) 归一化到 [0, 1]
        features = []
        for level, res in enumerate(self.resolutions):
            # 网格坐标
            grid_coords = x * res  # (N, 3)
            floor = grid_coords.long().clamp(0, res - 1)
            ceil = (floor + 1).clamp(max=res - 1)
            frac = grid_coords - floor.float()
            
            # 三线性插值的8个顶点
            corner_features = []
            for dx in [0, 1]:
                for dy in [0, 1]:
                    for dz in [0, 1]:
                        corner = torch.stack([
                            floor[:, 0] + dx,
                            floor[:, 1] + dy,
                            floor[:, 2] + dz
                        ], dim=-1)
                        # 哈希查找
                        h = self.spatial_hash(corner)
                        corner_feat = self.hash_tables[level][h]  # (N, feat_dim)
                        
                        # 插值权重
                        w_x = (1 - frac[:, 0]) if dx == 0 else frac[:, 0]
                        w_y = (1 - frac[:, 1]) if dy == 0 else frac[:, 1]
                        w_z = (1 - frac[:, 2]) if dz == 0 else frac[:, 2]
                        weight = (w_x * w_y * w_z).unsqueeze(-1)
                        
                        corner_features.append(corner_feat * weight)
            
            level_feat = sum(corner_features)
            features.append(level_feat)
        
        return torch.cat(features, dim=-1)  # (N, L * feat_dim)
```

### 2.3 整体网络

Instant-NGP 使用小型 MLP（1-2层）替代原始 NeRF 的8层MLP：

```
输入坐标(x,y,z)
    ↓
多分辨率哈希编码 → 高维特征
    ↓
小型MLP (2层, 隐藏层64维) → σ + 隐藏特征
    ↓
球谐编码(方向) + 小MLP → RGB
```

### 2.4 性能对比

| 方法 | 训练时间 | 渲染FPS | PSNR |
|------|----------|---------|------|
| NeRF | ~1-2天 | ~0.05 | 31.0 |
| Instant-NGP | ~5秒 | ~30 | 33.2 |

**200倍以上的训练加速！**

## 3. TensoRF (Tensor Radiance Fields)

### 3.1 核心思想

**TensoRF (Chen et al., SIGGRAPH 2022)** 将辐射场表示为**张量分解**，而非统一的MLP。

将3D辐射场看作一个4D张量 $\mathcal{V} \in \mathbb{R}^{X \times Y \times Z \times C}$，用CP或VM分解压缩：

### 3.2 CP 分解 (Canonical Polyadic)

$$\mathcal{V} \approx \sum_{r=1}^{R} \mathbf{a}_r \otimes \mathbf{b}_r \otimes \mathbf{c}_r$$

每个查询点的特征：

$$f(x,y,z) = \sum_{r=1}^{R} a_r(x) \cdot b_r(y) \cdot c_r(z)$$

其中 $a_r(x)$ 通过插值得到。

### 3.3 VM 分解 (Vector-Matrix)

$$\mathcal{V} \approx \sum_{r=1}^{R} \mathbf{M}_r \times_1 \mathbf{a}_r^{xy} \times_2 \mathbf{b}_r^{xz} \times_3 \mathbf{c}_r^{yz}$$

VM 分解比 CP 分解更紧凑，效果更好。

```python
class TensoRF(nn.Module):
    def __init__(self, grid_res=128, rank=24):
        super().__init__()
        self.rank = rank
        # CP分解的向量
        self.a_x = nn.Parameter(torch.randn(grid_res, rank) * 0.01)
        self.b_y = nn.Parameter(torch.randn(grid_res, rank) * 0.01)
        self.c_z = nn.Parameter(torch.randn(grid_res, rank) * 0.01)
        
        # 密度和颜色的分解（可不同）
        self.mat_xy = nn.Parameter(torch.randn(rank, grid_res, grid_res) * 0.01)
        self.mat_xz = nn.Parameter(torch.randn(rank, grid_res, grid_res) * 0.01)
        self.mat_yz = nn.Parameter(torch.randn(rank, grid_res, grid_res) * 0.01)
    
    def forward(self, x, d):
        """查询(x,y,z)处的密度和颜色"""
        # 插值得到每条轴的特征向量
        feat_x = grid_sample_1d(self.a_x, x[:, 0])  # (N, R)
        feat_y = grid_sample_1d(self.b_y, x[:, 1])
        feat_z = grid_sample_1d(self.c_z, x[:, 2])
        
        # CP密度
        density = (feat_x * feat_y * feat_z).sum(dim=-1)  # (N,)
        
        return F.relu(density), color
```

### 3.4 优势

- **存储高效**：分解大幅减少参数量
- **快速收敛**：直接优化体素而非隐式MLP
- **质量高**：PSNR 与 Instant-NGP 相当

## 4. Plenoxels (Radiance Fields without Neural Networks)

### 4.1 核心思想

**Plenoxels (Yu et al., CVPR 2022)** 完全**不用神经网络**，直接优化体素网格中的球谐系数。

### 4.2 表示

在3D体素网格的每个顶点存储球谐系数：

$$\mathbf{c}(\mathbf{x}, \mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm}(\mathbf{x}) \cdot Y_l^m(\mathbf{d})$$

- 稀疏正则化鼓励大部分体素为空
- 三线性插值获取连续值

### 4.3 性能

| 方法 | 训练时间 | PSNR | 是否有网络 |
|------|----------|------|-----------|
| NeRF | ~2天 | 31.0 | 是 |
| Plenoxels | ~11分钟 | 31.7 | 否 |

## 5. 方法对比总结

| 方法 | 核心思想 | 训练速度 | 渲染速度 | 质量 |
|------|----------|----------|----------|------|
| NeRF | MLP + 位置编码 | 极慢 | 极慢 | 基线 |
| Instant-NGP | 哈希编码 + 小MLP | 极快 | 快 | 更好 |
| TensoRF | 张量分解 | 快 | 快 | 相当 |
| Plenoxels | 纯体素优化 | 极快 | 快 | 相当 |

## 6. 其他重要变体

### 6.1 Mip-NeRF (2021)

使用**圆锥台 (cone)** 代替射线采样，解决抗锯齿问题：

$$\text{采样区域} = \text{Cone}(\mathbf{o}, \mathbf{d}, \text{pixel radius})$$

对每个圆锥台区域进行积分，自然支持多尺度渲染。

### 6.2 Mip-NeRF 360 (2022)

处理无界（360度）场景：
- **场景收缩**：将无限空间映射到有限区域
- **空间退耦**：前景和背景分开建模
- **位姿优化**：同时优化相机参数

### 6.3 Zip-NeRF (2023)

结合 Instant-NGP 的哈希编码和 Mip-NeRF 360 的抗锯齿：

$$\text{Zip-NeRF} = \text{Instant-NGP速度} + \text{Mip-NeRF 360质量}$$

成为当前单场景NeRF的最优方法之一。

### 6.4 Kilo-NeRF (2021)

将空间划分为多个小型NeRF，每个只负责一个小区域：
- 推理时根据查询位置选择对应的子网络
- 大幅减少每个子网络的参数量

## 7. 实用代码示例

### 7.1 使用 nerfstudio 框架

```bash
# 安装 nerfstudio
pip install nerfstudio

# 使用 Instant-NGP 训练
ns-train instant-ngp --data path/to/data

# 使用 Nerfacto（推荐基线）
ns-train nerfacto --data path/to/data

# 渲染视频
ns-render camera-path --load-config outputs/.../config.yml
```

### 7.2 使用 torch-ngp

```python
from torch_ngp.nerf import NeRFRenderer

renderer = NeRFRenderer(
    bound=1.0,          # 场景边界
    cuda_ray=True,      # CUDA光线追踪
    density_grid_res=128,
    min_near=0.1,
    density_thresh=1e-3,
)

# 训练
renderer.train(train_dataset, num_epochs=10)

# 渲染
images = renderer.render(poses, H, W, K)
```

## 8. 选择建议

| 场景 | 推荐方法 |
|------|----------|
| 高质量单场景 | Zip-NeRF / Mip-NeRF 360 |
| 快速实验 | Instant-NGP / nerfstudio |
| 无GPU环境 | Plenoxels |
| 大规模场景 | Block-NeRF / Mega-NeRF |
| 实时应用 | Instant-NGP |

---

**关键要点**：
1. Instant-NGP 用多分辨率哈希编码替代大型MLP，实现200倍训练加速
2. TensoRF 用张量分解压缩辐射场，同样实现高效训练
3. Plenoxels 证明完全不用神经网络也能达到NeRF质量
4. 当前最佳实践是 Zip-NeRF（结合哈希编码和抗锯齿）
