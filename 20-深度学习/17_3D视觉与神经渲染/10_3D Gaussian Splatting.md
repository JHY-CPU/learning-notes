# 10_3D Gaussian Splatting

## 1. 概述

**3D Gaussian Splatting (3DGS, Kerbl et al., SIGGRAPH 2023)** 是NeRF之后3D视觉领域最重要的突破。它用**3D高斯椭球**显式表示场景，结合**可微光栅化**实现实时高质量渲染。

### 1.1 与NeRF的核心区别

| 特性 | NeRF | 3DGS |
|------|------|------|
| 表示方式 | 隐式（MLP） | 显式（高斯点集合） |
| 渲染方式 | 体渲染（光线行进） | 光栅化（splatting） |
| 训练速度 | 数小时~数天 | 数分钟 |
| 渲染速度 | 数秒/帧 | 实时（100+ FPS） |
| 可编辑性 | 困难 | 容易（直接操作点） |

## 2. 3D高斯表示

### 2.1 每个高斯的参数

每个3D高斯由以下参数定义：

| 参数 | 符号 | 维度 | 说明 |
|------|------|------|------|
| 位置 | $\boldsymbol{\mu}$ | 3 | 中心坐标 $(x, y, z)$ |
| 协方差 | $\boldsymbol{\Sigma}$ | 6 | 3×3对称矩阵（6个自由度） |
| 不透明度 | $\alpha$ | 1 | 0~1之间 |
| 球谐系数 | $SH$ | 48 | 三阶球谐 (3+1)²×3 = 48 |

### 2.2 协方差矩阵的参数化

为保证协方差矩阵半正定，使用**缩放** $s \in \mathbb{R}^3$ 和**旋转** $q \in \mathbb{R}^4$（四元数）：

$$\boldsymbol{\Sigma} = R S S^T R^T$$

其中 $S = \text{diag}(s_x, s_y, s_z)$ 是缩放矩阵，$R$ 是旋转矩阵（由四元数 $q$ 转换）。

```python
import torch
import torch.nn as nn

def build_covariance(scaling, rotation):
    """
    从缩放和旋转构建协方差矩阵
    scaling: (N, 3)
    rotation: (N, 4) 四元数 (w, x, y, z)
    返回: (N, 3, 3)
    """
    # 归一化四元数
    rotation = rotation / (rotation.norm(dim=-1, keepdim=True) + 1e-8)
    
    # 四元数 → 旋转矩阵
    w, x, y, z = rotation.unbind(-1)
    R = torch.stack([
        1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y),
        2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x),
        2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)
    ], dim=-1).reshape(-1, 3, 3)
    
    # 缩放矩阵
    S = torch.diag_embed(scaling)  # (N, 3, 3)
    
    # 协方差 Σ = R S S^T R^T
    covariance = R @ S @ S.transpose(-1, -2) @ R.transpose(-1, -2)
    
    return covariance
```

### 2.3 球谐函数表示颜色

颜色由**球谐 (Spherical Harmonics)** 系数表示，视角相关：

$$c(\mathbf{d}) = \sum_{l=0}^{L} \sum_{m=-l}^{l} c_{lm} Y_l^m(\mathbf{d})$$

三阶球谐：$L = 3$，共 $(L+1)^2 = 16$ 个基函数，3个颜色通道 → 48维。

```python
def evaluate_sh(sh_coeff, directions):
    """
    从球谐系数和观察方向计算颜色
    sh_coeff: (N, 16, 3) 球谐系数
    directions: (N, 3) 观察方向（归一化）
    返回: (N, 3) 颜色
    """
    x, y, z = directions.unbind(-1)
    
    # 0阶
    sh = [0.28209479177387814 * torch.ones_like(x)]
    
    # 1阶
    sh.extend([
        -0.4886025119029199 * y,
        0.4886025119029199 * z,
        -0.4886025119029199 * x,
    ])
    
    # 2阶
    sh.extend([
        1.0925484305920792 * x * y,
        -1.0925484305920792 * y * z,
        0.31539156525252005 * (2*z*z - x*x - y*y),
        -1.0925484305920792 * x * z,
        0.5462742152960396 * (x*x - y*y),
    ])
    
    # 3阶（省略具体公式）
    # ...
    
    sh = torch.stack(sh, dim=-1)  # (N, 16)
    color = (sh.unsqueeze(-1) * sh_coeff).sum(dim=-2)  # (N, 3)
    return torch.sigmoid(color)  # 限制到 [0, 1]
```

## 3. 可微光栅化渲染

### 3.1 Splatting 流程

与 NeRF 的光线行进不同，3DGS 使用**前向投影**（splatting）：

```
3D高斯集合
    ↓ 投影到2D (视锥裁剪 + 仿射变换)
2D高斯椭圆
    ↓ 按深度排序 (画家算法)
排序后的2D高斯列表
    ↓ α混合
最终像素颜色
```

### 3.2 3D到2D投影

将3D高斯投影到2D屏幕空间。使用**仿射变换**的线性近似：

$$\boldsymbol{\Sigma}' = J W \boldsymbol{\Sigma} W^T J^T$$

其中 $W$ 是视图变换矩阵，$J$ 是投影变换的雅可比矩阵。

```python
def project_gaussians(means3d, covs3d, viewmat, projmat, H, W):
    """将3D高斯投影到2D"""
    # 视图变换
    means_cam = (viewmat[:3, :3] @ means3d.T + viewmat[:3, 3:]).T  # (N, 3)
    
    # 投影到屏幕
    means_h = torch.cat([means_cam, torch.ones(len(means_cam), 1)], dim=-1)
    means_clip = (projmat @ means_h.T).T
    means_screen = means_clip[:, :2] / means_clip[:, 3:4]  # (N, 2)
    
    # 协方差投影
    # J: 投影雅可比, W: 视图旋转
    J = compute_jacobian(means_cam, focal_x, focal_y)
    W = viewmat[:3, :3]
    cov2d = J @ W @ covs3d @ W.T @ J.T  # (N, 2, 2)
    
    return means_screen, cov2d
```

### 3.3 α混合渲染

对每个像素，按深度排序后进行前向α混合：

$$C(\mathbf{p}) = \sum_{i \in \mathcal{N}} c_i \alpha_i T_i \prod_{j=1}^{i-1}(1 - \alpha_j)$$

其中 $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$ 是累积透射率。

```python
def alpha_blend_render(sorted_colors, sorted_alphas):
    """前向α混合"""
    # sorted_colors: (N_pixels, K, 3) K个高斯按深度排序
    # sorted_alphas: (N_pixels, K)
    
    # 累积透射率
    T = torch.cumprod(1.0 - sorted_alphas, dim=-1)
    T = torch.cat([torch.ones_like(T[:, :1]), T[:, :-1]], dim=-1)
    
    # 加权混合
    weights = T * sorted_alphas  # (N_pixels, K)
    color = (weights.unsqueeze(-1) * sorted_colors).sum(dim=1)  # (N_pixels, 3)
    
    return color
```

### 3.4 2D高斯在像素中的权重

对于2D高斯 $g_i$ 在像素 $\mathbf{p}$ 处的贡献：

$$g_i(\mathbf{p}) = \exp\left(-\frac{1}{2}(\mathbf{p} - \boldsymbol{\mu}'_i)^T {\boldsymbol{\Sigma}'_i}^{-1} (\mathbf{p} - \boldsymbol{\mu}'_i)\right)$$

## 4. 完整3DGS模型

```python
class GaussianModel(nn.Module):
    def __init__(self, max_gaussians=100000):
        super().__init__()
        self.max_gaussians = max_gaussians
        
        # 可学习参数
        self.means = nn.Parameter(torch.zeros(max_gaussians, 3))
        self.sh_coeffs = nn.Parameter(torch.zeros(max_gaussians, 16, 3))
        self.scaling = nn.Parameter(torch.zeros(max_gaussians, 3))
        self.rotation = nn.Parameter(torch.zeros(max_gaussians, 4))
        self.opacity = nn.Parameter(torch.zeros(max_gaussians, 1))
        
        # 激活函数
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-8)
    
    def get_covariance(self):
        return build_covariance(
            self.scaling_activation(self.scaling),
            self.rotation_activation(self.rotation)
        )
    
    def forward(self, viewpoint_camera):
        """渲染一张图像"""
        cov3d = self.get_covariance()
        means2d, cov2d = project_gaussians(
            self.means, cov3d, 
            viewpoint_camera.world_view_transform,
            viewpoint_camera.projection_matrix,
            viewpoint_camera.H, viewpoint_camera.W
        )
        
        # 评估颜色（依赖视角）
        colors = evaluate_sh(self.sh_coeffs, view_dirs)
        alphas = self.opacity_activation(self.opacity)
        
        # 排序 + α混合
        rendered = alpha_blend_render(means2d, cov2d, colors, alphas)
        
        return rendered
```

## 5. 初始化

3DGS 通常从 **SfM 稀疏点云** 初始化：

1. 运行 COLMAP 获取稀疏点云和相机位姿
2. 每个SfM点初始化一个高斯
3. 球谐系数初始化为点的颜色（0阶）
4. 协方差初始化为各向同性小球

```python
def initialize_from_sfm(sfm_points, sfm_colors):
    """从SfM点云初始化高斯"""
    N = len(sfm_points)
    model = GaussianModel(max_gaussians=N * 2)  # 预留增长空间
    
    model.means.data[:N] = sfm_points
    model.sh_coeffs.data[:N, 0, :] = sfm_colors  # 0阶SH = 平均颜色
    model.scaling.data[:N] = torch.log(torch.tensor(0.01))  # 小球
    model.rotation.data[:N, 0] = 1.0  # 单位四元数 (w=1)
    model.opacity.data[:N] = inverse_sigmoid(0.5)  # 中等不透明度
    
    return model
```

## 6. 损失函数

$$\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{\text{D-SSIM}}$$

```python
def gaussian_splatting_loss(rendered, ground_truth, lambda_ssim=0.2):
    """3DGS损失函数"""
    l1_loss = F.l1_loss(rendered, ground_truth)
    
    # D-SSIM损失
    ssim_loss = 1.0 - ssim(rendered, ground_truth)
    
    return (1 - lambda_ssim) * l1_loss + lambda_ssim * ssim_loss
```

---

**关键要点**：
1. 3DGS用显式的3D高斯椭球表示场景，每个高斯有位置、协方差、不透明度、球谐颜色
2. 可微光栅化（splatting）替代光线行进，实现实时渲染
3. 协方差通过缩放+旋转参数化，保证半正定性
4. 球谐函数表示视角相关的颜色，支持镜面反射等效果
