# 41_神经辐射场 (NeRF) 基础原理

## 核心概念

- **NeRF（Neural Radiance Field）**：Mildenhall et al. (2020) 提出的新型3D场景表示方法，用一个MLP网络隐式编码3D场景的连续体积密度和颜色信息，通过体素渲染从任意新视角生成图像。
- **辐射场（Radiance Field）**：将3D场景表示为一个连续函数 $F_{\Theta}: (x, y, z, \theta, \phi) \to (R, G, B, \sigma)$，输入3D点坐标和观察方向，输出该点的颜色和体积密度。
- **体素渲染（Volume Rendering）**：沿相机光线对采样点的颜色和密度进行积分，生成2D图像。$\sigma$ 控制点的不透明度（密度），颜色沿光线累积。
- **位置编码（Positional Encoding）**：将输入坐标映射到高频傅里叶特征，使MLP能够学习场景的高频细节。$\gamma(p) = (\sin(2^0\pi p), \cos(2^0\pi p), \dots, \sin(2^{L-1}\pi p), \cos(2^{L-1}\pi p))$。
- **分层体积采样（Hierarchical Volume Sampling）**：使用两阶段策略——粗略网络在均匀采样点上预测密度分布，精细网络根据密度分布在重要性高的区域进行更密集的采样。
- **视角依赖（View Dependence）**：颜色输出依赖于观察方向，可以建模非朗伯表面（如镜面反射、高光）；密度 $\sigma$ 仅依赖于位置（与视角无关），保证几何一致性。

## 数学推导

**NeRF的辐射场表示：**
$$
F_{\Theta}(x, d) = (c, \sigma)
$$

其中 $x = (x, y, z)$ 是3D坐标，$d = (\theta, \phi)$ 是观察方向，$c = (r, g, b)$ 是颜色，$\sigma$ 是体积密度。

**体素渲染方程（光线颜色积分）：**
沿光线 $r(t) = o + td$ 的累积颜色：
$$
C(r) = \int_{t_n}^{t_f} T(t) \sigma(r(t)) c(r(t), d) dt
$$
$$
T(t) = \exp\left(-\int_{t_n}^{t} \sigma(r(s)) ds\right)
$$

其中 $T(t)$ 是累积透射率（从$t_n$到$t$的光线未被阻挡的概率）。

**离散化的体素渲染（数值积分）：**
将光线分为 $N$ 个区间，在每个区间内均匀采样：
$$
\hat{C}(r) = \sum_{i=1}^{N} T_i \cdot \alpha_i \cdot c_i
$$
$$
T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)
$$
$$
\alpha_i = 1 - \exp(-\sigma_i \delta_i)
$$

其中 $\delta_i = t_{i+1} - t_i$ 是采样间隔，$\alpha_i$ 是点 $i$ 的不透明度。

**训练损失（简单MSE）：**
$$
\mathcal{L} = \sum_{r \in \mathcal{R}} \|\hat{C}_c(r) - C(r)\|_2^2 + \|\hat{C}_f(r) - C(r)\|_2^2
$$

其中 $\hat{C}_c$ 和 $\hat{C}_f$ 分别是粗略网络和精细网络的预测颜色，$C(r)$ 是真实像素值。

## 直观理解

NeRF的核心思想是用一个"万能MLP"来记忆3D场景。你可以把NeRF想象成一个"3D场景的压缩器"——用一个小型神经网络（几MB）存储整个3D场景的信息。当你问"从某个角度看这个场景是什么样子"时，NeRF的计算过程是：

- 从相机位置向每个像素发射光线
- 沿光线采样多个3D点
- 对每个采样点，MLP预测该点"是什么颜色"和"是否在物体表面"
- 将所有采样点的信息按照"谁在前面谁遮挡后面"的规则进行合成，得到像素颜色

位置编码是关键——如果不加高频编码，MLP倾向于学习"平滑"的场景（就像模糊的图片），而高频编码让MLP可以表达锐利的边缘和精细的纹理。

## 代码示例

```python
import torch
import torch.nn as nn

class NeRF(nn.Module):
    """NeRF MLP (简化版)"""
    def __init__(self, pos_enc_dim=60, dir_enc_dim=24, hidden_dim=256):
        super().__init__()
        # 位置编码输入 -> 密度和中间特征
        self.block1 = nn.Sequential(
            nn.Linear(pos_enc_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        # 密度输出
        self.density_head = nn.Linear(hidden_dim, 1)
        # 中间特征 + 方向编码 -> 颜色
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + dir_enc_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),  # RGB颜色
            nn.Sigmoid(),  # 颜色在[0,1]范围内
        )

    def forward(self, x, d):
        # x: (N, pos_enc_dim) 位置编码后的3D坐标
        # d: (N, dir_enc_dim) 位置编码后的观察方向
        h = self.block1(x)
        sigma = self.density_head(h)
        h_dir = torch.cat([h, d], dim=-1)
        color = self.block2(h_dir)
        return color, sigma

def positional_encoding(points, num_frequencies=10):
    """位置编码: (x, y, z) -> sin/cos多频特征"""
    freq_bands = 2.0 ** torch.arange(0, num_frequencies)
    encoded = [points]
    for freq in freq_bands:
        encoded.append(torch.sin(points * freq))
        encoded.append(torch.cos(points * freq))
    return torch.cat(encoded, dim=-1)

# 模拟体素渲染
def volume_render(color_sigma_list, dists):
    """沿一条光线进行体素渲染 (简化)"""
    colors = torch.stack([c for c, _ in color_sigma_list])
    sigmas = torch.stack([s for _, s in color_sigma_list])
    alphas = 1 - torch.exp(-sigmas * dists)
    T = torch.cumprod(1 - alphas + 1e-10, dim=0)
    T = torch.cat([torch.ones_like(T[:1]), T[:-1]])
    weights = T * alphas
    rendered_color = (weights * colors).sum(dim=0)
    return rendered_color

# 测试 NeRF
model = NeRF()
x_encoded = positional_encoding(torch.randn(64, 3), 10)  # 64个采样点
d_encoded = positional_encoding(torch.randn(64, 3), 4)   # 方向编码
colors, densities = model(x_encoded, d_encoded)
print(f"颜色输出: {colors.shape}")
print(f"密度输出: {densities.shape}")
print(f"NeRF参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **3D视觉的革命性方法**：NeRF开创了"隐式神经场景表示（INR）"的新范式，颠覆了传统的显式3D表示（网格、体素、点云）。后续工作如Instant NGP（多分辨率哈希编码，训练提速100倍）、Mip-NeRF（抗锯齿）、NeRF-W（处理非静态场景）等不断改进其质量和效率。
- **从视图合成到3D重建**：NeRF最初用于新颖视图合成（给定一组照片生成新视角），后被扩展到3D重建（从NeRF中提取网格）、生成模型（EG3D、DreamFusion将NeRF与GAN或扩散模型结合）、SLAM（iMAP、NICE-SLAM）等广泛任务。
- **3D表示的演变**：NeRF的隐式表示方法代表了3D视觉从"显式几何表示"到"隐式函数表示"的重要转变，促进了可微渲染（differentiable rendering）技术的发展。3D Gaussian Splatting在此基础上提出更高效的显式表示方法。
