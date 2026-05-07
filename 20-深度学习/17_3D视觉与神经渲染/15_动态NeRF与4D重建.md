# 15_动态NeRF与4D重建

## 1. 动态场景建模

静态NeRF假设场景不变。**动态NeRF**将时间 $t$ 作为额外输入，建模随时间变化的场景：

$$F_\theta: (\mathbf{x}, \mathbf{d}, t) \rightarrow (\mathbf{c}, \sigma)$$

### 1.1 挑战

- 每个时间步需要独立的3D结构
- 需要保持时间连贯性
- 训练数据更稀疏（每时刻只有少量视角）

## 2. D-NeRF (Dynamic NeRF)

### 2.1 变形场思想

**D-NeRF (Pumarola et al., CVPR 2021)** 将动态建模为**从规范空间到观测空间的变形**：

$$\mathbf{x}' = \mathbf{x} + \Delta\mathbf{x} = \mathbf{x} + d_\phi(\mathbf{x}, t)$$

其中 $d_\phi$ 是**变形网络**，$d_\phi(\mathbf{x}, t)$ 输出从时间 $t$ 的位置 $\mathbf{x}$ 到规范空间位置的偏移。

### 2.2 网络结构

```
                  ┌─────────────────────────────┐
                  │      规范空间 (Canonical)      │
                  │    f_θ(x, d) → (c, σ)        │
                  └──────────────┬──────────────┘
                                 ↑
                              变形 x' = x + d_φ(x, t)
                                 ↑
                  ┌──────────────┴──────────────┐
                  │     观测空间 (Observation)     │
                  │      d_φ(x, t) → Δx          │
                  └─────────────────────────────┘
```

```python
class DNeRF(nn.Module):
    def __init__(self):
        super().__init__()
        # 变形网络：(x, y, z, t) → (Δx, Δy, Δz)
        self.deformation_net = nn.Sequential(
            nn.Linear(4 + 30, 256), nn.ReLU(),  # 3D位置 + 时间（位置编码后）
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 3)  # 位移
        )
        
        # 规范空间NeRF：(x', y', z', d) → (c, σ)
        self.canonical_nerf = NeRF(in_dim=63 + 27)  # 位置编码后
    
    def forward(self, x, d, t):
        # x: (B, 3), d: (B, 3), t: scalar or (B, 1)
        
        # 1. 编码时间
        t_enc = positional_encoding(t, L=10)  # 时间的位置编码
        
        # 2. 变形到规范空间
        x_with_t = torch.cat([x, t_enc], dim=-1)
        delta_x = self.deformation_net(x_with_t)
        x_canonical = x + delta_x
        
        # 3. 规范空间NeRF查询
        x_enc = positional_encoding(x_canonical, L=10)
        d_enc = positional_encoding(d, L=4)
        color, density = self.canonical_nerf(x_enc, d_enc)
        
        return color, density, delta_x
```

### 2.3 正则化

为保证变形的平滑性：

$$\mathcal{L}_{reg} = \lambda_1 \|\Delta\mathbf{x}\|^2 + \lambda_2 \|\nabla d_\phi\|^2$$

- 位移正则化：鼓励小的变形
- 梯度正则化：鼓励空间平滑的变形

## 3. Neural 3D Video Synthesis

### 3.1 HyperNeRF

**HyperNeRF (Park et al., ICCV 2021)** 处理拓扑变化（如手拿杯子和不拿杯子）：

将变形场扩展到**高维规范空间**：

$$\mathbf{x}' = (\mathbf{x} + d_\phi(\mathbf{x}, t), h_\psi(\mathbf{x}, t))$$

其中 $h_\psi$ 映射到高维空间，捕捉拓扑变化。

### 3.2 Neural Scene Flow Fields

**NSFF (Li et al., ICCV 2021)** 引入场景流：

$$\mathbf{x}' = \mathbf{x} + \mathbf{v}(\mathbf{x}, t) \cdot \Delta t$$

其中 $\mathbf{v}(\mathbf{x}, t)$ 是场景流场，预测每个点的3D速度。

## 4. 人体重建

### 4.1 Neural Body

**Neural Body (Peng et al., CVPR 2021)** 结合人体参数模型（SMPL）：

```
SMPL人体模型
    ↓
在SMPL顶点上放置潜在编码
    ↓
对每个查询点，查找最近顶点的编码
    ↓
神经网络解码 → 颜色、密度
```

```python
class NeuralBody(nn.Module):
    def __init__(self, smpl_model, num_vertices=6890, latent_dim=128):
        super().__init__()
        self.smpl = smpl_model
        
        # 每个顶点的潜在编码
        self.latent_codes = nn.Embedding(num_vertices, latent_dim)
        
        # 解码网络
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 3 + 27, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 4),  # (R, G, B, σ)
        )
    
    def forward(self, x, d, body_params):
        # 1. SMPL网格变形
        vertices, _ = self.smpl(body_params)
        
        # 2. 查询最近顶点
        dists = torch.cdist(x, vertices)  # (N, 6890)
        _, nearest_idx = dists.min(dim=-1)  # (N,)
        
        # 3. 获取潜在编码
        latent = self.latent_codes(nearest_idx)  # (N, 128)
        
        # 4. 解码
        x_enc = positional_encoding(x, L=10)
        d_enc = positional_encoding(d, L=4)
        feat = torch.cat([latent, x_enc, d_enc], dim=-1)
        output = self.decoder(feat)
        
        color = torch.sigmoid(output[:, :3])
        density = F.relu(output[:, 3])
        
        return color, density
```

### 4.2 Instant-NVR (Instant Neural Volumetric Humans)

结合 SMPL-X 和 Instant-NGP 的哈希编码，实现实时人体重建。

## 5. 4D生成

### 5.1 文本到4D

类似DreamFusion但增加时间维度：

```python
def train_text_to_4d(prompt, num_frames=16):
    """文本到4D生成"""
    # 初始化4D表示（动态NeRF）
    dynamic_nerf = DNeRF4D(num_frames=num_frames)
    
    for step in range(30000):
        # 随机采样时间
        t = random.randint(0, num_frames - 1)
        
        # 随机采样视角
        camera = sample_random_camera()
        
        # 渲染该时间步的视角
        rendered = dynamic_nerf.render(camera, t=t)
        
        # SDS损失（使用视频扩散模型或图像扩散模型）
        loss = sds_loss(rendered, prompt)
        
        # 时间一致性损失
        if t > 0:
            prev_rendered = dynamic_nerf.render(camera, t=t-1)
            loss += 0.1 * temporal_smoothness(rendered, prev_rendered)
        
        loss.backward()
        optimizer.step()
```

### 5.2 视频到4D

从视频重建动态3D场景：

```
视频帧序列 + 相机位姿
    ↓
逐帧3DGS重建
    ↓
时间维度建模（变形场/时序参数）
    ↓
4D高斯模型
```

## 6. 实时动态渲染

### 6.1 4D Gaussian Splatting

将时间维度融入3DGS：

```python
class DynamicGaussian4D(nn.Module):
    def __init__(self, base_gaussians, deformation_net):
        super().__init__()
        self.base = base_gaussians  # 规范空间的3DGS
        self.deform = deformation_net  # 变形网络
    
    def forward(self, camera, t):
        # 获取规范空间的高斯
        means = self.base.means
        covs = self.base.get_covariance()
        
        # 时间条件变形
        t_tensor = torch.tensor([t]).expand(len(means), 1)
        delta_means, delta_covs = self.deform(means, t_tensor)
        
        # 应用变形
        deformed_means = means + delta_means
        deformed_covs = covs + delta_covs
        
        # 渲染
        return rasterize(camera, deformed_means, deformed_covs, 
                        self.base.colors, self.base.opacities)
```

---

**关键要点**：
1. 动态NeRF通过变形场将观测空间映射到规范空间，处理时间变化
2. D-NeRF是最经典的动态NeRF，使用位移网络建模变形
3. 人体重建通常结合参数化模型（SMPL）提供强先验
4. 4D生成是当前热门方向，结合时间一致性正则化
