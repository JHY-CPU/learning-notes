# 12_从图像到3D：DreamFusion 与 SDS

## 1. 问题定义

**文本到3D生成 (Text-to-3D)**：给定文本描述（如 "一只穿着宇航服的猫"），生成对应的3D模型。

核心挑战：与2D图像生成不同，大规模的"文本-3D"配对数据几乎不存在。

## 2. Score Distillation Sampling (SDS)

### 2.1 核心思想

**DreamFusion (Poole et al., NeurIPS 2022)** 提出 SDS，巧妙地利用预训练的2D图像生成模型（扩散模型）作为3D生成的监督信号。

**关键洞察**：如果一个3D场景从所有角度看都像"一只穿宇航服的猫"，那它就是"一只穿宇航服的猫"。

### 2.2 SDS 损失

SDS 的梯度定义为：

$$\nabla_\theta \mathcal{L}_{\text{SDS}}(\theta) = \mathbb{E}_{t, \epsilon} \left[ w(t) \left( \epsilon_\phi(\mathbf{x}_t; t, y) - \epsilon \right) \frac{\partial \mathbf{x}}{\partial \theta} \right]$$

其中：
- $\theta$：3D表示的参数（如NeRF的参数）
- $\mathbf{x} = g(\theta)$：从3D表示渲染的2D图像
- $\epsilon_\phi$：预训练扩散模型的噪声预测
- $\epsilon \sim \mathcal{N}(0, I)$：采样的真实噪声
- $y$：文本条件
- $t$：扩散时间步

**直观理解**：SDS 梯度方向是"让渲染图像更像扩散模型认为正确的样子"。

### 2.3 代码实现

```python
import torch
import torch.nn.functional as F

class SDSLoss:
    def __init__(self, diffusion_model, guidance_scale=100):
        self.diffusion = diffusion_model  # 预训练的扩散模型 (如 Stable Diffusion)
        self.guidance_scale = guidance_scale
    
    def compute_sds_loss(self, rendered_image, text_prompt, theta):
        """
        计算SDS损失
        rendered_image: (1, 3, H, W) 从3D模型渲染的图像
        text_prompt: 文本描述
        theta: 3D模型参数（用于计算梯度）
        """
        B = rendered_image.shape[0]
        
        # 1. 随机采样时间步
        t = torch.randint(
            self.diffusion.num_timesteps // 4,  # 跳过太小的t
            self.diffusion.num_timesteps,
            (B,), device=rendered_image.device
        )
        
        # 2. 编码图像到潜空间
        with torch.no_grad():
            z_0 = self.diffusion.encode_image(rendered_image)  # (B, 4, h, w)
        
        # 3. 添加噪声
        epsilon = torch.randn_like(z_0)
        z_t = self.diffusion.add_noise(z_0, epsilon, t)
        
        # 4. 预测噪声
        epsilon_pred = self.diffusion.predict_noise(z_t, t, text_prompt)
        
        # 5. SDS梯度
        # ∇θ L_SDS = w(t) * (ε_φ(z_t; t, y) - ε) * ∂x/∂θ
        with torch.no_grad():
            noise_residual = epsilon_pred - epsilon
            # 通过扩散模型的解码器将噪声残差传回像素空间
            sds_grad = self.diffusion.decode_gradient(noise_residual)
        
        # 6. 应用梯度到渲染图像（链式法则自动传播到theta）
        loss = (rendered_image * sds_grad.detach()).sum()
        
        return loss
```

### 2.4 SDS 的工作流程

```
文本提示: "一只穿宇航服的猫"
    ↓
初始化3D模型 (随机NeRF)
    ↓
┌─────────────────────────┐
│ 随机选择一个视角          │
│ 渲染2D图像               │
│ 添加随机噪声 z_t          │
│ 扩散模型预测噪声 ε_φ      │
│ 计算SDS梯度              │
│ 反向传播更新3D模型        │
└─────────────────────────┘
    ↓ 重复 10K~15K 步
输出: 3D模型
```

## 3. DreamFusion 架构

### 3.1 3D表示

DreamFusion 使用 **Mip-NeRF 360** 作为3D表示：
- 处理无界场景
- 支持抗锯齿
- 效果好但训练较慢

### 3.2 引导模型

使用 **Imagen**（Google的扩散模型）作为2D先验：
- CLIP文本编码器
- U-Net噪声预测器
- 无分类器引导 (CFG)

### 3.3 训练技巧

| 技巧 | 说明 |
|------|------|
| 随机视角 | 每步从随机方位角和仰角渲染 |
| 相机距离随机化 | 避免只学某一距离 |
| 透视投影 | 使用真实相机模型 |
| 视图相关提示 | 在文本中添加 "front view"、"side view" 等 |

```python
def sample_random_camera():
    """随机采样相机位姿"""
    # 均匀采样方位角
    azimuth = torch.rand(1) * 2 * math.pi
    # 余弦加权采样仰角（偏好正面视角）
    elevation = torch.acos(1 - 2 * torch.rand(1))
    # 随机距离
    distance = torch.rand(1) * 2 + 1.5  # [1.5, 3.5]
    
    # 球坐标 → 笛卡尔坐标
    cam_pos = torch.tensor([
        distance * torch.sin(elevation) * torch.cos(azimuth),
        distance * torch.sin(elevation) * torch.sin(azimuth),
        distance * torch.cos(elevation)
    ])
    
    # look-at矩阵
    cam2world = look_at(cam_pos, target=torch.zeros(3))
    return cam2world
```

## 4. SDS 的变体

### 4.1 Variational Score Distillation (VSD)

**Prolific Dreamer (Wang et al., NeurIPS 2023)** 提出 VSD，改进 SDS 的质量：

$$\nabla_\theta \mathcal{L}_{\text{VSD}} = \mathbb{E} \left[ w(t) \left( \epsilon_\phi(\mathbf{x}_t) - \epsilon_\psi(\mathbf{x}_t) \right) \frac{\partial \mathbf{x}}{\partial \theta} \right]$$

区别：用另一个小的扩散模型 $\epsilon_\psi$ 替代真实噪声 $\epsilon$，更准确地估计分布差异。

### 4.2 Classifier Score Distillation (CSD)

使用分类器而非扩散模型引导：

$$\nabla_\theta \mathcal{L}_{\text{CSD}} = \mathbb{E}\left[\nabla_\mathbf{x} \log p(y|\mathbf{x}) \cdot \frac{\partial \mathbf{x}}{\partial \theta}\right]$$

### 4.3 SDS 改进对比

| 方法 | 引导信号 | 质量 | 速度 |
|------|----------|------|------|
| SDS (DreamFusion) | 扩散模型噪声预测 | 基线 | 快 |
| VSD (Prolific Dreamer) | 扩散 vs 小扩散 | 更高 | 慢 |
| SJC | 扩散模型分数 | 中等 | 中等 |

## 5. 3D一致性问题

### 5.1 Janus 问题

SDS 的最大问题是**多面脸 (Janus)**：3D模型从不同角度看时，面部朝向不一致。

原因：2D扩散模型只见过"正面脸"，缺乏"侧面脸"的约束。

### 5.2 解决方案

| 方法 | 思路 |
|------|------|
| 视图相关提示 | "front view of..." 、"side view of..." |
| 深度引导 | 同时引导深度图的一致性 |
| 粗到细 | 先学形状，再学纹理 |
| 正则化 | 深度平滑损失、法线一致性损失 |

```python
def depth_consistency_loss(rendered_depth, rendered_normals):
    """深度-法线一致性损失"""
    # 从深度图计算法线
    computed_normals = depth_to_normals(rendered_depth)
    # 法线方向一致性
    loss = F.l1_loss(rendered_normals, computed_normals)
    return loss
```

## 6. 代码实战

### 6.1 简化版训练循环

```python
def train_dreamfusion(text_prompt, num_steps=15000):
    """DreamFusion训练"""
    # 初始化3D模型
    nerf = MipNeRF360()
    optimizer = torch.optim.Adam(nerf.parameters(), lr=1e-3)
    
    # 加载扩散模型
    diffusion = load_stable_diffusion()
    sds_loss = SDSLoss(diffusion)
    
    for step in range(num_steps):
        # 1. 随机采样相机
        camera = sample_random_camera()
        
        # 2. 渲染
        rendered = nerf.render(camera)  # (1, 3, H, W)
        
        # 3. SDS损失
        loss = sds_loss.compute_sds_loss(rendered, text_prompt, nerf.parameters())
        
        # 4. 正则化
        if step % 500 == 0:
            # 体积正则化：鼓励紧凑
            loss += 0.1 * nerf.volume_regularization()
        
        # 5. 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 1000 == 0:
            print(f"Step {step}, SDS Loss: {loss.item():.4f}")
            # 可视化当前结果
            visualize_novel_view(nerf, camera)
```

## 7. 总结

| 方面 | 说明 |
|------|------|
| 核心创新 | SDS：用2D扩散模型的梯度引导3D生成 |
| 优势 | 不需要3D训练数据、可文本驱动 |
| 劣势 | 多视角不一致(Janus)、训练慢、质量上限 |
| 影响 | 开创了文本到3D的研究方向 |

---

**关键要点**：
1. SDS 的核心是将2D扩散模型的"知识"蒸馏到3D模型
2. SDS梯度 = 扩散模型预测噪声 - 真实噪声，方向是"更像目标"
3. Janus问题（多面脸）是SDS类方法的主要挑战
4. VSD通过引入辅助扩散模型改进了SDS的分布匹配
