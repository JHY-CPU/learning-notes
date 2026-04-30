# 15_扩散模型 (Diffusion Models) 物理直觉

## 核心概念
- **热力学类比**：扩散模型受非平衡热力学启发——前向过程像往咖啡中滴入牛奶（分子扩散），系统从有序走向无序（结构→噪声）；反向过程像从混乱中恢复秩序（去噪）。
- **前向扩散 (Forward Diffusion)**：逐步向数据添加高斯噪声，经过 $T$ 步后，数据被完全破坏为标准正态噪声 $x_T \sim \mathcal{N}(0, I)$。这个过程是固定的（无参数学习）。
- **反向去噪 (Reverse Denoising)**：学习一个神经网络 $p_\theta(x_{t-1}|x_t)$ 来逆转扩散过程——从纯噪声开始，逐步去噪恢复出数据。
- **马尔可夫链**：前向和反向过程都是马尔可夫链——当前状态只依赖于前一步状态，不依赖于更早的历史。
- **噪声条件得分网络 (Noise Conditional Score Network)**：扩散模型在数学上等价于学习数据分布的得分函数 $\nabla_x \log p(x)$（低概率密度区域→高概率密度区域的方向）。
- **逐步精化**：生成过程天然是迭代式的——从一幅纯噪声开始，每次去噪一点点，逐步揭示图像的结构。这与人类绘画（先勾轮廓再细化）有异曲同工之妙。

## 数学推导

**前向扩散过程**（方差调度 $\beta_1, \beta_2, ..., \beta_T$）：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} x_{t-1}, \beta_t I)
$$

任意时刻 $t$ 的解析形式（不需要迭代 $t$ 步）：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

**反向去噪过程**：

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

**物理直觉的数学表达**：

前向过程是熵增过程：数据的"信息"逐渐被噪声淹没

$$
H(x_t|x_0) = \frac{1}{2} \log(2\pi e (1 - \bar{\alpha}_t))
$$

当 $t \to T$ 时，$\bar{\alpha}_t \to 0$，条件熵最大，意味着 $x_t$ 几乎不包含 $x_0$ 的信息。

反向过程是"从混沌中创造秩序"：通过学习 $\nabla_x \log p_t(x)$（得分函数），引导粒子沿概率密度增加的方向移动，对应 Langevin 动力学的逆向。

## 直观理解
- **扩散模型 = 从碎纸机恢复文件**：前向过程像把文件放进碎纸机（不断添加噪声直到完全粉碎）。反向过程是学习如何把碎纸复原——一开始满桌纸屑（噪声），逐步拼凑出文档轮廓，最终得到完整文件。
- **物理类比：墨水在水中扩散**：前向过程是一滴墨水滴入清水（结构→均匀混合），反向过程是从均匀混合的水中"回收"墨水滴（去噪时把墨水分子聚拢回来）。
- **为什么用迭代而非一步生成**：一步生成就像试图从碎纸屑直接跳回完整文档——可能性太多，等于没有指导。逐步恢复的每一步只需要做微小的"猜对方向"修正，类似退火过程。
- **噪声调度 (Noise Schedule) 的作用**：噪声添加的速度（$\beta_t$ 的大小）决定了"信息破坏的速度"。太快则反向步骤太难学，太慢则需要很多步。常见的调度有线性调度（DDPM）和余弦调度（Improved DDPM）。

## 代码示例

```python
import torch
import torch.nn as nn
import math

def linear_beta_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    """线性噪声调度"""
    return torch.linspace(beta_start, beta_end, timesteps)

def cosine_beta_schedule(timesteps=1000, s=0.008):
    """余弦噪声调度（Improved DDPM）"""
    steps = torch.linspace(0, timesteps, timesteps + 1)
    f_t = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = f_t / f_t[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return torch.clip(betas, 0.0001, 0.9999)

class ForwardDiffusion:
    """前向扩散过程工具类"""
    def __init__(self, timesteps=1000, schedule='linear'):
        self.timesteps = timesteps
        
        if schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.betas = betas
        self.alphas = 1. - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def forward_process(self, x_0, t, noise=None):
        """
        一步计算任意时刻 t 的扩散结果
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1. - self.alpha_bars[t])[:, None, None, None]
        
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise
    
    def visualize_diffusion(self, x_0, steps=10):
        """可视化扩散过程：x_0 -> x_T"""
        all_steps = torch.linspace(0, self.timesteps - 1, steps, dtype=torch.long)
        x_t = x_0.clone()
        imgs = [x_0]
        for t in all_steps:
            noise = torch.randn_like(x_0)
            x_t, _ = self.forward_process(x_0, t.expand(x_0.size(0)), noise)
            imgs.append(x_t)
        return imgs  # 返回不同时间步的加噪结果

# 演示扩散过程
diffusion = ForwardDiffusion(timesteps=100, schedule='linear')

# 模拟一张"图像"（随机数据）
x_0 = torch.randn(1, 3, 32, 32)

# 逐步加噪观察
for t in [0, 10, 50, 90, 99]:
    x_t, noise = diffusion.forward_process(x_0, torch.tensor([t]))
    noise_level = 1 - diffusion.alpha_bars[t].item()
    print(f"t={t:3d}: 噪声比例={noise_level:.3f}, x_t 范围=[{x_t.min():.2f}, {x_t.max():.2f}]")

# 证明 x_T ~ N(0, I)
x_T, _ = diffusion.forward_process(x_0, torch.tensor([99]))
print(f"\nt=T 时:")
print(f"  x_T 均值: {x_T.mean().item():.4f} (应接近 0)")
print(f"  x_T 方差: {x_T.var().item():.4f} (应接近 1)")
print("\n前向扩散将数据完全破坏为标准正态噪声！")
```

## 深度学习关联
- **DDPM (Denoising Diffusion Probabilistic Models)**：首次证明了扩散模型能够生成媲美 GAN 的图像质量，通过重新参数化将去噪过程转化为"噪声预测"任务。
- **Score Matching + Langevin Dynamics**：Song & Ermon 的工作揭示了扩散模型与得分匹配的联系——训练一个网络预测 $\nabla_x \log p(x)$（得分函数），然后用 Langevin 采样生成。
- **SDE 统一视角**：Score SDE 论文将扩散模型和得分匹配统一为随机微分方程框架——前向是 SDE，反向是对应的逆时 SDE，扩散模型和得分匹配只是 SDE 框架的不同离散化方式。
- **Stable Diffusion 的工程突破**：通过将扩散过程从像素空间移到 VAE 潜空间（Latent Space），大幅降低了计算成本，使得扩散模型在消费级 GPU 上也能运行，引爆了文生图应用。
