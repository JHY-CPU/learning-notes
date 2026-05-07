# 16_场景表示与 SLAM

## 1. SLAM 概述

**SLAM (Simultaneous Localization and Mapping)** 是指在未知环境中，机器人/相机在移动的同时构建环境地图并定位自身位置。

### 1.1 经典SLAM vs 神经SLAM

| 特性 | 传统SLAM | 神经SLAM |
|------|----------|----------|
| 地图表示 | 稀疏点/稠密网格 | 隐式神经场 |
| 重建质量 | 稀疏或噪声大 | 高质量稠密 |
| 回环检测 | 基于特征匹配 | 学习型/特征型 |
| 全局一致性 | BA优化 | 隐式/显式优化 |
| 计算效率 | 高 | 较低 |

## 2. iMAP (Implicit Mapping and Positioning)

### 2.1 核心思想

**iMAP (Bloesch et al., 2021)** 首次用单一MLP作为场景表示，实现端到端SLAM：

```
实时RGB-D帧
    ↓
相机位姿估计 (优化位姿)
    ↓
场景更新 (优化NeRF参数)
    ↓
联合优化位姿 + 场景
```

### 2.2 关键设计

```python
class iMAP:
    def __init__(self):
        # 场景表示：单个MLP
        self.scene_network = NeRF()
        # 相机位姿
        self.keyframe_poses = []  # SE(3)位姿
        self.keyframe_images = []
    
    def tracking(self, new_frame):
        """追踪：固定场景，优化位姿"""
        pose = self.keyframe_poses[-1].clone()
        pose.requires_grad_(True)
        optimizer = torch.optim.Adam([pose], lr=1e-3)
        
        for _ in range(20):
            rendered = render(self.scene_network, pose, intrinsics)
            loss = compute_loss(rendered, new_frame)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return pose.detach()
    
    def mapping(self, selected_keyframes):
        """建图：固定位姿，优化场景"""
        optimizer = torch.optim.Adam(self.scene_network.parameters(), lr=1e-4)
        
        for _ in range(100):
            # 随机采样像素
            pixels = sample_random_pixels(selected_keyframes)
            
            loss = 0
            for kf_idx, pixel_coords in pixels:
                pose = self.keyframe_poses[kf_idx]
                rays = generate_rays(pixel_coords, pose, intrinsics)
                rendered_color, rendered_depth = render_rays(self.scene_network, rays)
                gt_color, gt_depth = get_gt(kf_idx, pixel_coords)
                loss += F.mse_loss(rendered_color, gt_color)
                loss += F.mse_loss(rendered_depth, gt_depth)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def select_key_pixels(self):
        """选择信息丰富的像素（自监督采样策略）"""
        # 渲染当前帧，找深度误差大的区域
        # 优先采样这些区域的像素来更新场景
        rendered_depth = render_depth(self.scene_network, current_pose)
        gt_depth = get_depth(current_frame)
        error = (rendered_depth - gt_depth).abs()
        # 概率加权采样
        prob = error / error.sum()
        selected = torch.multinomial(prob, num_samples=1024)
        return selected
```

### 2.3 局限

- 单MLP难以扩展到大场景
- 位姿估计精度有限
- 易受动态物体干扰

## 3. NICE-SLAM

### 3.1 层次化场景表示

**NICE-SLAM (Zhu et al., CVPR 2022)** 用**多分辨率网格特征**替代单MLP：

```
粗分辨率网格 (32³): 全局结构
    ↓ 三线性插值
中分辨率网格 (64³): 中等细节
    ↓ 三线性插值
细分辨率网格 (128³): 精细细节
    ↓
共享解码MLP → 颜色、SDF
```

### 3.2 实现

```python
class NICESLAM(nn.Module):
    def __init__(self, grid_sizes=[32, 64, 128], feat_dims=[32, 32, 8]):
        super().__init__()
        # 多分辨率特征网格
        self.feat_grids = nn.ModuleList([
            nn.Parameter(torch.randn(1, fd, gs, gs, gs) * 0.01)
            for gs, fd in zip(grid_sizes, feat_dims)
        ])
        
        # 解码MLP
        total_feat = sum(feat_dims)
        self.geo_decoder = nn.Sequential(
            nn.Linear(total_feat, 64), nn.ReLU(),
            nn.Linear(64, 1)  # SDF
        )
        self.color_decoder = nn.Sequential(
            nn.Linear(total_feat + 3 + 6, 64), nn.ReLU(),  # 特征 + 位置编码 + 法线
            nn.Linear(64, 3), nn.Sigmoid()
        )
    
    def query(self, points, normals=None):
        """查询点的SDF和颜色"""
        features = []
        for feat_grid in self.feat_grids:
            # 三线性插值采样特征
            feat = F.grid_sample(
                feat_grid,
                points.reshape(1, 1, 1, -1, 3),  # 归一化到 [-1, 1]
                align_corners=True
            ).reshape(-1, feat_grid.shape[1])
            features.append(feat)
        
        concat_feat = torch.cat(features, dim=-1)
        sdf = self.geo_decoder(concat_feat)
        
        if normals is not None:
            x_enc = positional_encoding(points, L=6)
            color_input = torch.cat([concat_feat, x_enc, normals], dim=-1)
            color = self.color_decoder(color_input)
        else:
            color = None
        
        return sdf, color
```

### 3.3 与iMAP对比

| 特性 | iMAP | NICE-SLAM |
|------|------|-----------|
| 场景表示 | 单MLP | 多分辨率特征网格 + 小MLP |
| 可扩展性 | 差 | 好 |
| 重建质量 | 中 | 高 |
| 内存 | 小 | 大 |
| 回环检测 | 无 | 有 |

## 4. 神经SLAM的损失函数

### 4.1 几何损失

```python
def sdf_loss(sdf_pred, depth_gt, truncation=0.1):
    """截断SDF损失"""
    # 从深度计算伪SDF
    sdf_gt = depth_to_sdf(depth_gt, ray_directions)
    
    # 截断范围内的损失
    mask = torch.abs(sdf_gt) < truncation
    loss = F.l1_loss(sdf_pred[mask], sdf_gt[mask])
    
    # 空间区域SDF=1，表面区域接近0
    return loss

def eikonal_loss(gradients):
    """Eikonal损失：确保SDF梯度范数为1"""
    return ((gradients.norm(dim=-1) - 1) ** 2).mean()
```

### 4.2 颜色损失

```python
def photometric_loss(rendered_rgb, gt_rgb):
    """光度损失"""
    l1 = F.l1_loss(rendered_rgb, gt_rgb)
    ssim_loss = 1.0 - compute_ssim(rendered_rgb, gt_rgb)
    return 0.8 * l1 + 0.2 * ssim_loss
```

## 5. 3DGS-SLAM

### 5.1 高斯Splatting SLAM

**GS-SLAM** 系列工作将3DGS作为场景表示，实现实时SLAM：

```
RGB-D帧输入
    ↓
追踪 (定位): 渲染+对齐
    ↓
建图 (高斯更新): 新区域致密化
    ↓
回环检测与全局优化
```

### 5.2 SplaTAM

**SplaTAM (Keetha et al., CVPR 2024)** 是代表性工作：

```python
class SplaTAM:
    def __init__(self):
        self.gaussians = GaussianModel()
        self.poses = []
    
    def track(self, frame):
        """基于渲染的追踪"""
        pose = self.poses[-1].clone()
        pose.requires_grad_(True)
        optimizer = torch.optim.Adam([pose], lr=1e-3)
        
        for _ in range(100):
            rendered = self.gaussians.render(pose, self.intrinsics)
            
            # 颜色损失
            color_loss = F.l1_loss(rendered['rgb'], frame['rgb'])
            # 深度损失
            depth_loss = F.l1_loss(rendered['depth'], frame['depth'])
            
            loss = color_loss + 0.5 * depth_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return pose.detach()
    
    def map(self, frame, pose):
        """建图：更新高斯"""
        for _ in range(50):
            rendered = self.gaussians.render(pose, self.intrinsics)
            loss = gaussian_splatting_loss(rendered, frame)
            loss.backward()
            self.optimizer.step()
        
        # 致密化
        self.densify(frame, pose)
```

## 6. 大规模场景

### 6.1 场景分割

大场景需要分块处理：

```python
class LargeScaleSLAM:
    def __init__(self, chunk_size=50.0):
        self.chunk_size = chunk_size
        self.chunks = {}  # (ix, iy, iz) → NeRF/Gaussian模型
    
    def get_chunk(self, position):
        """根据位置获取对应的场景块"""
        ix = int(position[0] // self.chunk_size)
        iy = int(position[1] // self.chunk_size)
        iz = int(position[2] // self.chunk_size)
        key = (ix, iy, iz)
        
        if key not in self.chunks:
            self.chunks[key] = GaussianModel()
        
        return self.chunks[key]
```

### 6.2 块间重叠

相邻场景块之间需要重叠区域保证连续性：

```
块1: [0, 50]  块2: [40, 90]  块3: [80, 130]
        重叠区域: [40, 50], [80, 90]
```

## 7. 实用工具

### 7.1 ORB-SLAM3

```python
import orbslam3

# ORB-SLAM3 初始化
slam = orbslam3.System(
    vocab_path="ORBvoc.txt",
    config_path="config.yaml",
    sensor_type=orbslam3.Sensor.MONOCULAR
)

# 追踪
for frame in video_stream:
    pose = slam.process_image_mono(frame, timestamp)
    if slam.is_tracking():
        # 获取地图点
        map_points = slam.get_tracked_map_points()
```

---

**关键要点**：
1. 神经SLAM用隐式神经场替代传统稀疏地图，实现高质量稠密重建
2. iMAP开创了单NeRF做SLAM的先河，但难以扩展
3. NICE-SLAM用多分辨率特征网格解决了可扩展性问题
4. 3DGS-SLAM结合了高质量渲染和实时性能，是当前最优方向
