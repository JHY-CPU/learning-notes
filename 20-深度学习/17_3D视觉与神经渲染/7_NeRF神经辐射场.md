# 7_NeRF：神经辐射场

## 1. NeRF 概述

**NeRF (Neural Radiance Fields, Mildenhall et al., ECCV 2020)** 是3D视觉领域最具影响力的工作之一。它用一个MLP网络隐式地表示3D场景，能够从一组2D图像合成高质量的新视角。

### 1.1 核心思想

给定一组已知相机位姿的图像，学习一个连续的5D函数：

$$F_\theta: (\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$$

- $\mathbf{x} = (x, y, z)$：3D空间位置
- $\mathbf{d} = (\theta, \phi)$：观察方向
- $\mathbf{c} = (r, g, b)$：颜色
- $\sigma$：体积密度（volume density）

## 2. 体渲染 (Volume Rendering)

### 2.1 连续体渲染方程

NeRF 的渲染基于**体渲染**理论。沿光线 $\mathbf{r}(t) = \mathbf{o} + t\mathbf{d}$，颜色为：

$$C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \cdot \sigma(\mathbf{r}(t)) \cdot \mathbf{c}(\mathbf{r}(t), \mathbf{d}) \, dt$$

其中**透射率 (Transmittance)**：

$$T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s)) \, ds\right)$$

**物理含义**：
- $T(t)$：光线从近平面到 $t$ 处未被任何粒子遮挡的概率
- $\sigma(\mathbf{r}(t))$：在 $t$ 处存在粒子的密度
- $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$：从方向 $\mathbf{d}$ 观察时该粒子的颜色

### 2.2 离散化（数值积分）

实际计算中使用**分层采样**，将光线分为 $N$ 段：

$$\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot \mathbf{c}_i$$

其中：
- $t_i \sim \mathcal{U}\left[t_n + \frac{(i-1)(t_f-t_n)}{N}, t_n + \frac{i(t_f-t_n)}{N}\right]$（均匀采样）
- $\delta_i = t_{i+1} - t_i$（相邻采样点间距）
- $\alpha_i = 1 - \exp(-\sigma_i \cdot \delta_i)$（不透明度）
- $T_i = \prod_{j=1}^{i-1}(1 - \alpha_j)$（累积透射率）

```python
def volume_rendering(colors, densities, t_vals, dirs):
    """
    体渲染
    colors: (N_rays, N_samples, 3)
    densities: (N_rays, N_samples, 1)
    t_vals: (N_rays, N_samples)
    返回: (N_rays, 3)
    """
    # 计算相邻采样点间距
    delta = t_vals[:, 1:] - t_vals[:, :-1]  # (N_rays, N_samples-1)
    # 末尾加一个大值
    delta = torch.cat([delta, torch.full_like(delta[:, :1], 1e10)], dim=-1)
    
    # 乘以方向范数
    delta = delta * torch.norm(dirs, dim=-1, keepdim=True)
    
    # 不透明度
    alpha = 1.0 - torch.exp(-densities * delta)  # (N_rays, N_samples)
    
    # 累积透射率
    # T_i = prod_{j<i} (1 - alpha_j)
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([
        torch.ones_like(transmittance[:, :1]),
        transmittance[:, :-1]
    ], dim=-1)
    
    # 权重
    weights = transmittance * alpha  # (N_rays, N_samples)
    
    # 加权颜色求和
    rgb = (weights.unsqueeze(-1) * colors).sum(dim=1)  # (N_rays, 3)
    
    # 深度图（加权平均深度）
    depth = (weights * t_vals).sum(dim=1)  # (N_rays,)
    
    return rgb, depth, weights
```

## 3. 位置编码

### 3.1 傅里叶编码

NeRF 使用傅里叶位置编码将坐标映射到高维空间：

$$\gamma(p) = \left(\sin(2^0\pi p), \cos(2^0\pi p), \ldots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p)\right)$$

- 位置 $\mathbf{x}$：$L=10$，输出 $3 + 3 \times 2 \times 10 = 63$ 维
- 方向 $\mathbf{d}$：$L=4$，输出 $3 + 3 \times 2 \times 4 = 27$ 维

```python
def positional_encoding(x, L):
    """傅里叶位置编码"""
    out = [x]
    for i in range(L):
        out.append(torch.sin(2**i * torch.pi * x))
        out.append(torch.cos(2**i * torch.pi * x))
    return torch.cat(out, dim=-1)
```

## 4. NeRF 网络架构

### 4.1 框架

```
输入: 位置(x,y,z) → 位置编码(63维) → MLP → σ + 隐特征
                                                  ↓
输入: 方向(θ,φ) → 位置编码(27维) → 拼接 → MLP → RGB
```

### 4.2 完整实现

```python
import torch
import torch.nn as nn

class NeRF(nn.Module):
    def __init__(self, pos_L=10, dir_L=4, hidden=256):
        super().__init__()
        # 位置编码后的维度
        pos_enc_dim = 3 + 3 * 2 * pos_L   # 63
        dir_enc_dim = 3 + 3 * 2 * dir_L   # 27
        
        # 前半部分：位置 → σ + 特征
        self.layers_before = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        
        # 残差连接
        self.layers_after = nn.Sequential(
            nn.Linear(hidden + pos_enc_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        
        # 密度头
        self.density_head = nn.Linear(hidden, 1)
        
        # 颜色头
        self.color_layer = nn.Sequential(
            nn.Linear(hidden + dir_enc_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 3), nn.Sigmoid(),
        )
        
        self.pos_L = pos_L
        self.dir_L = dir_L
    
    def forward(self, pos, dirs):
        # pos: (B, 3), dirs: (B, 3)
        pos_enc = positional_encoding(pos, self.pos_L)  # (B, 63)
        dir_enc = positional_encoding(dirs, self.dir_L)  # (B, 27)
        
        h = self.layers_before(pos_enc)           # (B, 256)
        h = self.layers_after(torch.cat([h, pos_enc], dim=-1))  # (B, 256)
        
        sigma = self.density_head(h)               # (B, 1)
        sigma = F.relu(sigma)  # 密度必须非负
        
        h_color = torch.cat([h, dir_enc], dim=-1)  # (B, 256+27)
        color = self.color_layer(h_color)           # (B, 3)
        
        return color, sigma
```

## 5. 视角相关颜色

### 5.1 为什么需要方向输入

物体表面的外观取决于观察方向：
- **漫反射**：颜色与方向无关
- **镜面反射**：颜色强烈依赖方向（高光）
- **菲涅尔效应**：边缘与中心的反射率不同

NeRF 通过在**最后阶段**引入方向依赖来建模这些效果，而密度 $\sigma$ 只与位置有关（几何是确定的，不因观察方向改变）。

## 6. 体积密度

### 6.1 密度的含义

$\sigma(\mathbf{x})$ 表示在位置 $\mathbf{x}$ 存在粒子的密度（类似CT扫描中的衰减系数）。

- $\sigma$ 大：该位置不透明，光线被阻挡
- $\sigma = 0$：该位置透明，光线穿过

密度通过 ReLU 激活保证非负。

### 6.2 密度与不透明度的关系

$$\alpha = 1 - \exp(-\sigma \cdot \delta)$$

- $\sigma$ 大或 $\delta$ 大 → $\alpha \to 1$（完全不透明）
- $\sigma$ 小或 $\delta$ 小 → $\alpha \approx \sigma \cdot \delta$（近似线性）

## 7. 损失函数

NeRF 使用简单的**均方误差损失**：

$$\mathcal{L} = \sum_{\mathbf{r} \in \mathcal{R}} \left\|\hat{C}(\mathbf{r}) - C_{gt}(\mathbf{r})\right\|^2$$

其中 $\hat{C}$ 是渲染颜色，$C_{gt}$ 是真实像素颜色。

同时优化**粗网络**和**细网络**：

$$\mathcal{L} = \mathcal{L}_{coarse} + \mathcal{L}_{fine}$$

## 8. 渲染流程

```python
def render_image(model, H, W, K, pose, near=0.1, far=10.0, N_samples=64):
    """渲染整张图像"""
    # 生成光线
    j, i = torch.meshgrid(torch.arange(H, dtype=torch.float32),
                          torch.arange(W, dtype=torch.float32))
    # 像素坐标 → 归一化相机坐标
    dirs = torch.stack([
        (i - K[0,2]) / K[0,0],
        -(j - K[1,2]) / K[1,1],
        -torch.ones_like(i)
    ], dim=-1)  # (H, W, 3)
    
    # 相机坐标 → 世界坐标
    dirs = dirs @ pose[:3, :3].T
    origins = pose[:3, 3].expand(dirs.shape)
    
    # 沿光线采样
    t_vals = torch.linspace(near, far, N_samples)
    
    # 展平处理
    rays_o = origins.reshape(-1, 3)
    rays_d = dirs.reshape(-1, 3)
    
    # 查询网络、体渲染
    rgb, depth, weights = render_rays(model, rays_o, rays_d, t_vals)
    
    return rgb.reshape(H, W, 3), depth.reshape(H, W)
```

---

**关键要点**：
1. NeRF 将3D场景编码为一个5D函数 $F(\mathbf{x}, \mathbf{d}) \rightarrow (\mathbf{c}, \sigma)$
2. 体渲染公式是核心：通过沿光线积分颜色和密度来渲染像素
3. 位置编码解决了MLP的频谱偏差，使网络能表示高频细节
4. 密度只与位置有关，颜色与方向有关——几何确定性
