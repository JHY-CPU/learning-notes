# 8_NeRF 训练与渲染详解

## 1. 光线生成

### 1.1 从相机参数生成光线

每条光线由**原点** $\mathbf{o}$ 和**方向** $\mathbf{d}$ 定义：

```python
def get_rays(H, W, K, c2w):
    """
    生成每条光线的原点和方向
    H, W: 图像尺寸
    K: 内参 (3, 3)
    c2w: 相机到世界变换 (4, 4)
    """
    # 像素网格
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing='xy'
    )
    
    # 像素坐标 → 相机坐标（归一化）
    dirs = torch.stack([
        (i - K[0, 2]) / K[0, 0],   # x = (u - cx) / fx
        -(j - K[1, 2]) / K[1, 1],  # y = -(v - cy) / fy
        -torch.ones_like(i)          # z = -1（朝-z方向看）
    ], dim=-1)  # (H, W, 3)
    
    # 相机坐标 → 世界坐标
    rays_d = torch.sum(dirs.unsqueeze(-2) * c2w[:3, :3], dim=-1)  # (H, W, 3)
    rays_o = c2w[:3, 3].expand(rays_d.shape)                      # (H, W, 3)
    
    return rays_o, rays_d
```

### 1.2 光线公式

$$\mathbf{r}(t) = \mathbf{o} + t \cdot \mathbf{d}$$

## 2. 分层采样 (Hierarchical Sampling)

### 2.1 粗采样 (Coarse Sampling)

沿光线均匀采样 $N_c$ 个点：

$$t_i = t_n + \frac{i}{N_c}(t_f - t_n), \quad i = 0, 1, \ldots, N_c - 1$$

```python
def sample_coarse(rays_o, rays_d, near, far, N_samples):
    """粗采样：均匀采样"""
    # 在 [near, far] 之间均匀采样
    t_vals = torch.linspace(near, far, N_samples)
    # 添加随机扰动（训练时）
    if training:
        mids = 0.5 * (t_vals[1:] + t_vals[:-1])
        upper = torch.cat([mids, t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids])
        t_vals = lower + (upper - lower) * torch.rand(N_samples)
    
    # 计算采样点坐标
    pts = rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * t_vals.unsqueeze(-1)
    # pts: (N_rays, N_samples, 3)
    
    return pts, t_vals
```

### 2.2 精细采样 (Fine Sampling)

粗网络的权重 $w_i$ 指示了物体存在的位置。在权重大的区域分配更多采样点：

```python
def sample_fine(rays_o, rays_d, t_coarse, weights, N_fine):
    """
    精细采样：基于粗网络权重的逆CDF采样
    t_coarse: (N_rays, N_coarse) 粗采样的t值
    weights: (N_rays, N_coarse) 粗网络的渲染权重
    """
    # 归一化权重为概率分布
    weights = weights + 1e-5  # 防止0
    pdf = weights / weights.sum(dim=-1, keepdim=True)
    
    # CDF
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], dim=-1)  # (N_rays, N_coarse+1)
    
    # 逆CDF采样
    u = torch.rand(N_rays, N_fine)
    # 在CDF上插值
    indices = torch.searchsorted(cdf, u, right=True)
    below = (indices - 1).clamp(0)
    above = indices.clamp(max=cdf.shape[-1] - 1)
    
    # 线性插值
    denom = cdf.gather(1, above) - cdf.gather(1, below)
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t_fine = t_coarse.gather(1, below) + \
             (u - cdf.gather(1, below)) / denom * \
             (t_coarse.gather(1, above) - t_coarse.gather(1, below))
    
    return t_fine
```

### 2.3 完整采样流程

```
粗采样 (N_c=64个均匀点)
    ↓
粗网络推理 → 权重
    ↓
逆CDF采样 (N_f=128个重点采样点)
    ↓
合并排序去重
    ↓
细网络推理 → 最终渲染
```

## 3. 权重计算详解

### 3.1 渲染权重

每个采样点的权重决定了它对最终像素颜色的贡献：

$$w_i = T_i \cdot \alpha_i = T_i \cdot (1 - \exp(-\sigma_i \delta_i))$$

其中累积透射率：

$$T_i = \prod_{j=1}^{i-1}(1 - \alpha_j) = \exp\left(-\sum_{j=1}^{i-1}\sigma_j \delta_j\right)$$

```python
def compute_weights(densities, t_vals, dirs):
    """
    计算渲染权重
    densities: (N_rays, N_samples, 1)
    t_vals: (N_rays, N_samples)
    """
    # 计算delta
    delta = t_vals[:, 1:] - t_vals[:, :-1]
    delta = torch.cat([delta, torch.full_like(delta[:, :1], 1e10)], dim=-1)
    delta = delta * torch.norm(dirs, dim=-1, keepdim=True)
    
    # 不透明度
    alpha = 1.0 - torch.exp(-densities.squeeze(-1) * delta)
    
    # 累积透射率
    transmittance = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    transmittance = torch.cat([
        torch.ones_like(transmittance[:, :1]),
        transmittance[:, :-1]
    ], dim=-1)
    
    # 权重
    weights = transmittance * alpha
    
    return weights  # (N_rays, N_samples)
```

## 4. 训练流程

### 4.1 训练伪代码

```python
def train_nerf(images, poses, K, near, far, num_epochs=200000):
    """NeRF训练流程"""
    coarse_net = NeRF()
    fine_net = NeRF()
    optimizer = torch.optim.Adam(
        list(coarse_net.parameters()) + list(fine_net.parameters()),
        lr=5e-4
    )
    # 学习率衰减
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99999)
    
    for step in range(num_epochs):
        # 1. 随机选取一张图像和一批光线
        img_idx = random.randint(0, len(images) - 1)
        ray_indices = random.sample(range(H * W), N_rays)
        
        # 2. 生成光线
        rays_o, rays_d = get_rays(H, W, K, poses[img_idx])
        rays_o = rays_o.reshape(-1, 3)[ray_indices]
        rays_d = rays_d.reshape(-1, 3)[ray_indices]
        target = images[img_idx].reshape(-1, 3)[ray_indices]
        
        # 3. 粗采样 + 粗网络
        pts_c, t_c = sample_coarse(rays_o, rays_d, near, far, N_c=64)
        colors_c, densities_c = coarse_net(pts_c, rays_d)
        rgb_c, weights_c = volume_rendering(colors_c, densities_c, t_c, rays_d)
        
        # 4. 精细采样 + 细网络
        t_f = sample_fine(rays_o, rays_d, t_c, weights_c, N_f=128)
        t_all = torch.sort(torch.cat([t_c, t_f], dim=-1), dim=-1)[0]
        pts_f = rays_o + rays_d * t_all.unsqueeze(-1)
        colors_f, densities_f = fine_net(pts_f, rays_d)
        rgb_f, _, _ = volume_rendering(colors_f, densities_f, t_all, rays_d)
        
        # 5. 损失
        loss = F.mse_loss(rgb_c, target) + F.mse_loss(rgb_f, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
```

### 4.2 训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| Batch rays | 1024 | 每次采样的光线数 |
| $N_c$ | 64 | 粗采样点数 |
| $N_f$ | 128 | 精细采样点数 |
| 学习率 | $5 \times 10^{-4}$ | 指数衰减 |
| 迭代次数 | 200K~500K | 取决于场景复杂度 |
| 训练时间 | ~1-2天 | 单GPU |

## 5. 推理与新视角合成

### 5.1 渲染新视角

```python
@torch.no_grad()
def render_novel_view(model, H, W, K, novel_pose, chunk=1024):
    """渲染新视角"""
    rays_o, rays_d = get_rays(H, W, K, novel_pose)
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    
    # 分批渲染防止OOM
    all_rgb = []
    for i in range(0, len(rays_o), chunk):
        ro = rays_o[i:i+chunk]
        rd = rays_d[i:i+chunk]
        rgb, _, _ = render_rays(model, ro, rd, near, far)
        all_rgb.append(rgb)
    
    rgb_image = torch.cat(all_rgb, dim=0).reshape(H, W, 3)
    return (rgb_image.clamp(0, 1) * 255).to(torch.uint8)
```

## 6. 训练技巧

### 6.1 位姿估计

训练NeRF需要精确的相机位姿。通常使用COLMAP进行SfM重建获取位姿。

### 6.2 背景处理

- **白色/黑色背景**：添加背景颜色项
- **真实背景**：使用Alpha合成
- **随机背景**：训练时用随机颜色替换，提升泛化

```python
# 带背景的体渲染
def render_with_background(colors, densities, t_vals, dirs, bg_color):
    rgb, depth, weights = volume_rendering(colors, densities, t_vals, dirs)
    acc_alpha = weights.sum(dim=-1, keepdim=True)  # 累积不透明度
    rgb_final = rgb + (1 - acc_alpha) * bg_color
    return rgb_final
```

### 6.3 近远平面估计

- 自动估计：从稀疏点云获取 $[t_n, t_f]$
- 经验设置：室内场景 $[0.1, 10.0]$，室外 $[1.0, 100.0]$

## 7. 计算瓶颈

### 7.1 为什么NeRF训练慢

1. **逐光线独立处理**：无法利用空间连贯性
2. **大量网络查询**：每条光线 ~192次查询（64+128）
3. **双重网络**：粗网络 + 细网络

### 7.2 性能分析

| 操作 | 占比 |
|------|------|
| MLP前向传播 | ~80% |
| 位置编码 | ~10% |
| 体渲染 | ~5% |
| 数据加载 | ~5% |

---

**关键要点**：
1. 分层采样是NeRF效率的关键：粗网络定位物体，细网络精细渲染
2. 渲染权重 $w_i = T_i \alpha_i$ 实现了体积密度到像素颜色的可微映射
3. 训练时随机采样光线（而非整张图像），显著降低了内存需求
4. NeRF 的主要瓶颈是 MLP 查询次数太多，这是后续加速工作的出发点
