# 11_3D Gaussian 的优化与致密化

## 1. 优化总览

3DGS 的训练是一个**自适应**过程：不仅优化现有高斯的参数，还会动态地**添加和删除**高斯，以自适应地覆盖场景的各个区域。

### 1.1 优化流程

```
初始化 (SfM点云)
    ↓
渲染 + 计算损失
    ↓
反向传播更新参数
    ↓
致密化 (Clone/Split + Split/Clone)
    ↓
剪枝 (去除低贡献高斯)
    ↓
重复迭代 (30K步)
```

## 2. 梯度计算

### 2.1 可微光栅化的梯度

α混合渲染中，颜色对各参数的梯度：

$$\frac{\partial C}{\partial c_i} = w_i \quad \text{(颜色梯度 = 渲染权重)}$$

$$\frac{\partial C}{\partial \alpha_i} = c_i T_i - \frac{1}{1-\alpha_i}(C - C_{<i})$$

其中 $C_{<i}$ 是前面所有高斯的累积贡献。

### 2.2 协方差的梯度

通过链式法则，从2D协方差梯度回传到缩放和旋转：

$$\frac{\partial \mathcal{L}}{\partial s_k} = \sum_{ij} \frac{\partial \mathcal{L}}{\partial \Sigma'_{ij}} \frac{\partial \Sigma'_{ij}}{\partial s_k}$$

```python
class DifferentiableRasterizer(torch.autograd.Function):
    """可微光栅化器（简化版）"""
    
    @staticmethod
    def forward(ctx, means2d, cov2d, colors, alphas, image_size):
        # 按深度排序
        sorted_idx = torch.argsort(depths)
        sorted_colors = colors[sorted_idx]
        sorted_alphas = alphas[sorted_idx]
        
        # 为每个像素收集前K个高斯
        # 计算每个高斯对每个像素的贡献
        # α混合得到最终颜色
        
        # 保存中间变量用于反向传播
        ctx.save_for_backward(weights, alphas, sorted_idx)
        
        return rendered_image
    
    @staticmethod
    def backward(ctx, grad_output):
        weights, alphas, sorted_idx = ctx.saved_tensors
        # 计算各参数的梯度
        grad_colors = ...
        grad_alphas = ...
        grad_means = ...
        grad_cov = ...
        return grad_means, grad_cov, grad_colors, grad_alphas, None
```

## 3. 自适应致密化

### 3.1 致密化的触发条件

每隔一定迭代步（通常每100步），检查每个高斯的状态：

**欠重建区域**：梯度大但高斯太小或太稀疏
- 条件：平均梯度 > 阈值 且 高斯尺寸 < 场景平均尺寸的一定比例

**过重建区域**：高斯太大
- 条件：高斯尺寸 > 场景平均尺寸的一定比例

### 3.2 Clone（克隆）

对于**欠重建**的小高斯：创建一个副本，位置添加随机扰动。

$$\boldsymbol{\mu}_{new} = \boldsymbol{\mu}_{old} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma^2)$$

```python
def clone_gaussians(model, grad_threshold, scene_extent):
    """克隆小梯度区域的高斯"""
    # 统计梯度
    avg_grad = model.means.grad.norm(dim=-1)
    
    # 条件：梯度大 + 高斯尺寸小
    is_high_grad = avg_grad > grad_threshold
    is_small = model.scaling.exp().max(dim=-1)[0] < 0.01 * scene_extent
    
    to_clone = is_high_grad & is_small
    
    # 克隆
    new_means = model.means[to_clone] + torch.randn_like(model.means[to_clone]) * 0.01
    new_sh = model.sh_coeffs[to_clone].clone()
    new_scaling = model.scaling[to_clone].clone()
    new_rotation = model.rotation[to_clone].clone()
    new_opacity = model.opacity[to_clone].clone()
    
    # 追加到模型
    model.means = nn.Parameter(torch.cat([model.means, new_means], dim=0))
    # ... 其他参数 ...
    
    return to_clone.sum().item()  # 返回新增数量
```

### 3.3 Split（分裂）

对于**过重建**的大高斯：将一个大高斯分裂为两个较小的高斯。

```python
def split_gaussians(model, grad_threshold, scene_extent):
    """分裂大高斯"""
    avg_grad = model.means.grad.norm(dim=-1)
    
    is_high_grad = avg_grad > grad_threshold
    is_large = model.scaling.exp().max(dim=-1)[0] > 0.01 * scene_extent
    
    to_split = is_high_grad & is_large
    
    # 分裂为两个，缩放减半
    n_split = to_split.sum()
    split_means = model.means[to_split]
    split_scaling = model.scaling[to_split] - torch.log(torch.tensor(2.0))
    
    # 两个新高斯分别偏移
    offset = torch.randn(n_split, 3) * 0.01
    means_1 = split_means + offset
    means_2 = split_means - offset
    
    # 合并
    new_means = torch.cat([means_1, means_2], dim=0)
    new_scaling = split_scaling.repeat(2, 1)
    # ... 其他参数类似 ...
    
    return n_split * 2
```

### 3.4 剪枝 (Pruning)

定期移除贡献极低的高斯：

```python
def prune_gaussians(model, opacity_threshold=0.005, size_threshold=None):
    """剪枝低贡献高斯"""
    # 不透明度过低的高斯
    keep = torch.sigmoid(model.opacity.squeeze()) > opacity_threshold
    
    # 过大的高斯（在特定尺度以上）
    if size_threshold:
        max_scale = model.scaling.exp().max(dim=-1)[0]
        keep &= max_scale < size_threshold
    
    # 应用剪枝
    model.means = nn.Parameter(model.means[keep])
    model.sh_coeffs = nn.Parameter(model.sh_coeffs[keep])
    model.scaling = nn.Parameter(model.scaling[keep])
    model.rotation = nn.Parameter(model.rotation[keep])
    model.opacity = nn.Parameter(model.opacity[keep])
    
    return keep.sum().item()  # 返回剩余数量
```

## 4. 优化策略

### 4.1 学习率调度

| 参数 | 初始学习率 | 最终学习率 | 衰减方式 |
|------|-----------|-----------|----------|
| 位置 $\boldsymbol{\mu}$ | $1.6 \times 10^{-4}$ | $1.6 \times 10^{-6}$ | 指数衰减 |
| 球谐 $SH$ | $2.5 \times 10^{-3}$ | $2.5 \times 10^{-5}$ | 指数衰减 |
| 不透明度 $\alpha$ | $5.0 \times 10^{-2}$ | $5.0 \times 10^{-3}$ | 指数衰减 |
| 缩放 $s$ | $5.0 \times 10^{-3}$ | $5.0 \times 10^{-4}$ | 指数衰减 |
| 旋转 $q$ | $1.0 \times 10^{-3}$ | $1.0 \times 10^{-4}$ | 指数衰减 |

### 4.2 优化器

使用 **Adam** 优化器，$\beta_1=0.9, \beta_2=0.999$。

```python
def setup_optimizer(model):
    """设置优化器"""
    params = [
        {'params': [model.means], 'lr': 1.6e-4, 'name': 'means'},
        {'params': [model.sh_coeffs], 'lr': 2.5e-3, 'name': 'sh'},
        {'params': [model.opacity], 'lr': 5e-2, 'name': 'opacity'},
        {'params': [model.scaling], 'lr': 5e-3, 'name': 'scaling'},
        {'params': [model.rotation], 'lr': 1e-3, 'name': 'rotation'},
    ]
    return torch.optim.Adam(params)
```

### 4.3 高斯数量变化

典型训练过程中的高斯数量变化：

```
初始化:     ~5K  (SfM点)
7K步:      ~30K
15K步:     ~100K
30K步:     ~300K ~ 500K
```

## 5. 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| PSNR | $10 \log_{10}\frac{MAX^2}{MSE}$ | 峰值信噪比 |
| SSIM | 结构相似性 | 感知质量 |
| LPIPS | 深度特征距离 | 感知质量（学习型） |

## 6. 实战代码

### 6.1 完整训练循环

```python
def train_3dgs(model, optimizer, train_cameras, train_images, num_iters=30000):
    """3DGS训练循环"""
    for iteration in range(num_iters):
        # 随机选择一个视角
        cam_idx = random.randint(0, len(train_cameras) - 1)
        camera = train_cameras[cam_idx]
        gt_image = train_images[cam_idx]
        
        # 渲染
        rendered = model(camera)
        
        # 损失
        loss = gaussian_splatting_loss(rendered, gt_image)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 记录梯度（用于致密化）
        if model.means.grad is not None:
            model.means_gradient_accum += model.means.grad.norm(dim=-1)
            model.denom += 1
        
        optimizer.step()
        
        # 致密化 (每100步)
        if iteration % 100 == 0 and iteration < 15000:
            grad_threshold = 2e-4
            clone_gaussians(model, grad_threshold, scene_extent)
            split_gaussians(model, grad_threshold, scene_extent)
            prune_gaussians(model)
            # 重置梯度累积
            model.means_gradient_accum.zero_()
            model.denom.zero_()
        
        # 定期剪枝 (每1000步)
        if iteration % 1000 == 0:
            prune_gaussians(model, opacity_threshold=0.005)
```

### 6.2 使用 gsplat 库

```python
import gsplat

# 渲染
colors, alphas, info = gsplat.rasterization(
    means=means3d,
    quats=rotations,
    scales=scalings,
    opacities=opacities,
    colors=sh_colors,
    viewmats=viewmat,
    Ks=intrinsics,
    width=W, height=H,
    render_mode="RGB"
)

# 自动微分支持反向传播
loss = F.mse_loss(colors, gt_image)
loss.backward()
```

## 7. 内存与效率

### 7.1 内存占用

| 高斯数量 | 参数内存 | 显存（渲染） |
|----------|---------|-------------|
| 100K | ~50MB | ~2GB |
| 500K | ~250MB | ~6GB |
| 1M | ~500MB | ~10GB |

### 7.2 加速技巧

1. **分块渲染**：将图像分块处理，减少排序开销
2. **视锥裁剪**：只渲染视锥内的高斯
3. **快速排序**：GPU上的基数排序
4. **量化**：压缩球谐系数精度

---

**关键要点**：
1. 致密化是3DGS成功的关键：自适应增加高斯覆盖场景细节
2. Clone用于填补稀疏区域，Split用于细化粗糙区域
3. 不同参数使用不同学习率，位置学习率最低
4. gsplat 库提供了高效的可微光栅化实现
