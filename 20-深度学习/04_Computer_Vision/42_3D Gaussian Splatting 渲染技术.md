# 42_3D Gaussian Splatting 渲染技术

## 核心概念

- **3D Gaussian Splatting (3DGS)**：Kerbl et al. (2023) 提出的实时神经渲染方法，使用数百万个3D高斯椭球体显式表示3D场景，通过可微光栅化实现高质量的新颖视图合成。
- **显式表示 vs 隐式表示**：与NeRF的隐式MLP表示不同，3DGS用显式的3D高斯点云表示场景，可以像点云一样操作和编辑，渲染速度达到实时（>30 FPS）。
- **3D高斯的参数**：每个高斯包含：位置 $\mu \in \mathbb{R}^3$（中心点）、协方差矩阵 $\Sigma \in \mathbb{R}^{3\times3}$（控制椭球的形状和方向）、不透明度 $\alpha \in \mathbb{R}$、球谐函数系数（控制视角依赖的颜色）。
- **可微光栅化（Differentiable Rasterization）**：将3D高斯投影到2D图像平面的快速算法，支持梯度反向传播以优化高斯参数。使用分块排序（tile sorting）和扩展可见性算法实现高效渲染。
- **自适应密度控制（Adaptive Density Control）**：在训练过程中自动增加（克隆/分裂高方差区域的高斯）或删除（低不透明度/体积过大的高斯）高斯点，动态调整场景表示的复杂度。
- **快速训练**：3DGS从SfM（Structure from Motion）点云初始化，使用可微光栅化进行端到端优化，训练时间通常为30分钟到1小时（NeRF的1/10）。

## 数学推导

**3D高斯的定义：**
$$
G(x) = \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)\right)
$$

其中 $\Sigma$ 是协方差矩阵，控制高斯椭球的形状和方向。$\Sigma$ 通过旋转矩阵 $R$ 和缩放矩阵 $S$ 参数化以保证正定性：
$$
\Sigma = R S S^T R^T
$$

**3D到2D的投影：**
给定视角变换矩阵 $W$，投影到图像平面的2D协方差矩阵为：
$$
\Sigma' = J W \Sigma W^T J^T
$$

其中 $J$ 是投影变换的雅可比矩阵。

**体素渲染（按序混合）：**
沿光线的颜色累积与NeRF类似，但按深度排序后的高斯进行$\alpha$混合：
$$
C = \sum_{i=1}^N c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)
$$

其中 $\alpha_i$ 是投影后2D高斯在像素位置的值乘以学习到的不透明度。

**损失函数：**
$$
\mathcal{L} = (1 - \lambda) \mathcal{L}_1 + \lambda \mathcal{L}_{D-SSIM}
$$

其中 $\mathcal{L}_1$ 是L1色差，$\mathcal{L}_{D-SSIM}$ 是结构相似性损失（SSIM），$\lambda = 0.2$。

## 直观理解

3D Gaussian Splatting可以理解为"用大量小椭球拼出3D场景"。每个椭球（3D高斯）有三个关键属性：在哪里（位置）、多大的方向（形状和旋转）、是什么颜色（球谐系数）。

渲染时，这些椭球被"拍扁"投影到2D图像上，然后按从远到近的顺序混合——就像在一张画布上从后往前叠加油画颜料。由于这个过程是"光栅化"（类似传统图形管线的渲染方式），速度远快于NeRF的光线追踪。

这种显式表示的优势是直观和可控：想删除场景中的某个物体？直接删除对应的高斯即可。想移动物体？移动对应的高斯即可。不像NeRF那样"信息锁定在MLP权重中"难以编辑。

## 代码示例

```python
import torch
import torch.nn as nn

class GaussianScene(nn.Module):
    """3D Gaussian Splatting 场景表示 (简化)"""
    def __init__(self, num_gaussians=100000):
        super().__init__()
        # 位置
        self.means = nn.Parameter(torch.randn(num_gaussians, 3) * 2)
        # 缩放 (log空间确保正数)
        self.scales = nn.Parameter(torch.randn(num_gaussians, 3) - 2)
        # 旋转 (四元数)
        self.quats = nn.Parameter(torch.randn(num_gaussians, 4))
        # 不透明度 (logit空间)
        self.opacities = nn.Parameter(torch.randn(num_gaussians))
        # 颜色: SH系数 (简化: 只用0阶, 即漫反射颜色)
        self.sh_coeffs = nn.Parameter(torch.rand(num_gaussians, 3))

    def get_covariance(self, idx):
        """构建协方差矩阵 (简化)"""
        scale = torch.exp(self.scales[idx])
        # 标准化四元数
        quat = self.quats[idx] / (self.quats[idx].norm() + 1e-8)
        # 四元数转旋转矩阵 (简化)
        R = torch.eye(3)  # 真实实现需要四元数转换
        S = torch.diag(scale)
        Sigma = R @ S @ S.T @ R.T
        return Sigma

    def render(self, camera_matrix):
        """简化渲染 (实际需要光栅化管线)"""
        # 获取所有高斯的投影到屏幕坐标
        opacity = torch.sigmoid(self.opacities)
        colors = torch.sigmoid(self.sh_coeffs)
        # 排序 (按深度)
        depths = self.means @ camera_matrix[:3, 2]  # 简化的深度计算
        sorted_idx = torch.argsort(depths)
        # alpha混合 (简化: 忽略协方差影响)
        rendered = torch.zeros(3)
        transmittance = 1.0
        for idx in sorted_idx:
            alpha = opacity[idx]
            rendered += transmittance * alpha * colors[idx]
            transmittance *= (1 - alpha)
            if transmittance < 0.01:
                break
        return rendered

# 创建场景
scene = GaussianScene(1000)
print(f"高斯位置: {scene.means.shape}")
print(f"高斯的参数总量: {sum(p.numel() for p in scene.parameters())}")

# 模拟渲染一个像素
cam_pose = torch.eye(4)
pixel_color = scene.render(cam_pose)
print(f"渲染像素颜色: {pixel_color}")

# 与NeRF对比
print("\n3DGS vs NeRF 核心差异:")
print("- 表示: 显式(高斯点云) vs 隐式(MLP权重)")
print("- 渲染: 光栅化 vs 光线追踪")
print("- 速度: 实时(>30FPS) vs 慢(几秒/帧)")
print("- 编辑: 可编辑(增删改高斯) vs 难编辑(权重难解释)")
```

## 深度学习关联

- **实时神经渲染的突破**：3DGS在渲染质量接近NeRF的同时，实现了实时帧率（30-100+FPS），使神经渲染从"离线渲染"走向"实时交互"，推动了VR/AR、数字孪生等应用的发展。
- **显示表示的优势回归**：在NeRF隐式表示主导了几年后，3DGS证明了精心设计的显式表示可以在质量和效率上全面超越隐式方法，引发了3D表示的新一轮探索（如SuGaR、Mip-Splatting等改进工作）。
- **多模态3D内容创建**：3DGS的显式高斯表示天然适合与扩散模型结合（如DreamGaussian、GaussianDreamer），用于从文本/图像生成3D内容，以及用于动态场景重建（4D Gaussian Splatting）等前沿方向。
