# 59_BEV (Bird's Eye View) 视角转换技术

## 核心概念

- **BEV（Bird's Eye View，鸟瞰图）**：将多视角相机或LiDAR的感知结果统一转换到俯视视角的网格表示中，使自动驾驶系统可以"从上方看"整个驾驶场景。
- **BEV感知的优势**：在BEV空间中，物体尺度一致（近大远小的问题消失）、时间对齐方便（不同帧在BEV空间中可以直接对齐）、决策规划更直观（路径规划在BEV坐标中进行）。
- **视角转换（View Transformation）**：从图像视角到BEV视角的映射是BEV感知的核心挑战。传统方法基于几何投影（IPM），深度学习方法通过可微投影或注意力机制实现。
- **IPM（Inverse Perspective Mapping）**：利用相机内外参，将前视图像素投影到地平面（假设平坦地面）得到BEV图。缺点是对地面起伏敏感，且无法处理垂直方向的信息。
- **基于深度估计的转换**：LSS（Lift-Splat-Shoot）方法——为每个像素预测深度分布（"lift"），将所有像素投影到3D空间，再"spalt"到BEV网格上。需要明确预测深度。
- **基于注意力/Transformer的转换**：BEVFormer使用可变形注意力（Deformable Attention）将BEV Query与多视角图像特征关联，通过迭代Refinement学习从图像到BEV的映射，无需显式深度估计。

## 数学推导

**IPM（逆向透视映射）：**
给定相机内参 $K$ 和外参 $[R|t]$，从图像坐标 $(u, v)$ 到地平面 $Z=0$ 上的世界坐标 $(X, Y, 0)$ 的映射：
$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R | t] \begin{bmatrix} X \\ Y \\ 0 \\ 1 \end{bmatrix}
$$

当 $Z=0$ 时，这个映射可以简化为一个 $3\times3$ 的单应矩阵 $H$。

**LSS（Lift-Splat-Shoot）的深度分布预测：**
对每个像素 $(u, v)$ 预测 $D$ 个离散深度值的概率分布：
$$
\alpha_{(u,v)} \in \mathbb{R}^D, \quad \sum_{d=1}^D \alpha_{(u,v), d} = 1
$$

"Lift"——将每个像素的 $C$ 维特征提升到3D空间：$f_{(u,v,d)} = \alpha_{(u,v), d} \cdot f_{(u,v)}$

"Splat"——将所有3D点汇聚到BEV网格中（通过pillar pooling）。

**BEVFormer的可变形注意力：**
BEV Query $Q_{p}$（对应BEV网格位置 $p$）通过可变形注意力从多视角图像特征 $F$ 中聚合信息：
$$
\text{DeformAttn}(Q_p, p, F) = \sum_{i=1}^{N_{ref}} \sum_{j=1}^{N_{points}} A_{ij} \cdot F_i(p + \Delta p_{ij})
$$

其中 $N_{ref}$ 是参考视角数，$N_{points}$ 是每个参考点的采样点数（通常4个），$A_{ij}$ 是注意力权重，$\Delta p_{ij}$ 是采样偏移。

**BEV空间到图像空间的投影：**
对于BEV网格坐标 $(x, y)$ 和高度 $z$（可学习），通过相机参数投影到图像坐标：
$$
p = P(x, y, z)
$$

其中 $P$ 是3D到2D的投影函数（由相机内外参决定）。

## 直观理解

BEV视角转换可以理解为将多个"装在车身上的摄像头"拍摄的图像，"缝合"成一张从车顶向下看的"地图"。图像像是从人眼高度看出去的——有透视、有遮挡、远处物体变小。BEV像是从卫星上看的——所有物体以统一的尺度呈现，车辆的位置关系一目了然。

传统方法IPM像是"把地面上的贴纸揭下来贴到俯视图上"——假设地面是平的，只把地面上的内容投影到BEV。但当路面有坡度、颠簸时，IPM会产生畸变。

深度学习方法（LSS、BEVFormer）不需要假设地面是平的——LSS通过预测深度信息把每个像素"放置"到3D空间的准确位置（"lift"），再"拍"到BEV平面（"splat"）。BEVFormer则更进一步，通过注意力机制让模型"学会"如何将图像特征映射到BEV空间，不需要显式深度估计。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSSViewTransformer(nn.Module):
    """Lift-Splat-Shoot 视角转换 (简化版)"""
    def __init__(self, in_channels=64, out_channels=64, 
                 num_depth_bins=41, bev_size=(200, 200)):
        super().__init__()
        self.num_depth_bins = num_depth_bins
        self.bev_h, self.bev_w = bev_size
        
        # 深度分布预测
        self.depth_net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_depth_bins, 1),
        )
        # BEV特征编码
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, cam_params):
        """
        x: (B, C, H, W) 图像特征
        cam_params: 相机参数 (用于Lift-Splat的投影矩阵)
        """
        B, C, H, W = x.shape
        
        # 预测深度分布
        depth_logits = self.depth_net(x)  # (B, D, H, W)
        depth_probs = F.softmax(depth_logits, dim=1)  # (B, D, H, W)
        
        # Lift: 将图像特征提升到3D
        # (B, C, H, W) × (B, D, H, W) -> (B, C*D, H, W)
        x_lift = x.unsqueeze(1) * depth_probs.unsqueeze(2)
        x_lift = x_lift.reshape(B, C * self.num_depth_bins, H, W)
        
        # Splat: 简化版 - 投影到BEV (真实实现需要基于相机参数的投影)
        # 此处假设已经将特征投影到BEV空间
        x_bev = F.adaptive_avg_pool2d(x_lift, (self.bev_h, self.bev_w))
        
        return self.bev_encoder(x_bev)

class BEVFormerLayer(nn.Module):
    """BEVFormer 的可变形注意力层 (简化版)"""
    def __init__(self, embed_dim=256, num_heads=8, num_points=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_points = num_points
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, bev_query, img_features, ref_points):
        """
        bev_query: (B, N_bev, D) BEV Query
        img_features: (B, N_img, D) 图像特征 (展平)
        ref_points: (B, N_bev, 2) BEV网格参考点在图像上的投影坐标
        """
        # 简化: 用参考点的投影坐标通过插值采样图像特征
        B, N_bev, D = bev_query.shape
        # 模拟采样 (实际是grid_sample)
        sampled_features = img_features[:, :N_bev, :]  # (B, N_bev, D)
        
        # 注意力
        attn_out, _ = self.attn(bev_query, sampled_features, sampled_features)
        return self.norm(bev_query + attn_out)

# 多视角BEV融合
class MultiViewBEVFusion(nn.Module):
    """多视角相机BEV融合"""
    def __init__(self, num_views=6, feat_dim=64, bev_size=(200, 200)):
        super().__init__()
        self.num_views = num_views
        self.feat_dim = feat_dim
        self.bev_h, self.bev_w = bev_size
        
        # 每个视角的独立LSS
        self.view_transformers = nn.ModuleList([
            LSSViewTransformer(feat_dim, feat_dim) for _ in range(num_views)
        ])
        self.fusion_conv = nn.Conv2d(feat_dim * num_views, feat_dim, 1)

    def forward(self, multi_view_feats, cam_params_list):
        # multi_view_feats: list of (B, C, H, W) × num_views
        bev_list = []
        for i in range(self.num_views):
            bev = self.view_transformers[i](
                multi_view_feats[i], cam_params_list[i]
            )
            bev_list.append(bev)
        # 拼接各视角BEV特征
        bev_cat = torch.cat(bev_list, dim=1)
        # 融合
        bev_fused = self.fusion_conv(bev_cat)
        return bev_fused  # (B, C, bev_h, bev_w)

# 测试
lss = LSSViewTransformer(in_channels=64, out_channels=64)
x = torch.randn(2, 64, 32, 64)  # 单视图特征
bev_out = lss(x, None)
print(f"BEV特征: {bev_out.shape}")

multi_fusion = MultiViewBEVFusion(num_views=6, feat_dim=64)
views = [torch.randn(2, 64, 32, 64) for _ in range(6)]
bev_fused = multi_fusion(views, [None]*6)
print(f"多视角BEV融合: {bev_fused.shape}")  # (2, 64, 200, 200)

print("\nBEV感知的关键挑战:")
print("- 视角转换: 从图像空间到BEV空间的精确映射")
print("- 多视角融合: 6个相机视野的无缝拼接")
print("- 时间融合: 历史帧的BEV特征与当前帧对齐")
print("- 动态物体: 在BEV空间中处理运动目标")
```

## 深度学习关联

- **自动驾驶感知的主流范式**：BEV感知已成为现代自动驾驶系统的主流方案（Tesla、Waymo、百度Apollo等）。BEVFormer通过将图像特征直接转换到BEV空间，统一了多视角感知，再在此基础上进行检测、分割、跟踪等任务。
- **从BEV到4D占用网格**：BEV感知正在向更丰富的3D/4D占用网格（Occupancy Network）演变——不仅在地平面上表示物体，还在整个3D空间中进行体素化的语义占用预测，更好地处理悬空物体和不规则形状。
- **端到端自动驾驶的基础**：BEV表示是实现"端到端自动驾驶"（从感知直接到规划）的关键中间表示。UniAD等端到端模型在BEV空间中联合处理感知-预测-规划，展示了从原始传感器到控制命令的全可微学习路径。
