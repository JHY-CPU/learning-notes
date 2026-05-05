# 35_扩散模型的 SDE 视角解释

## 核心概念

- **SDE 统一框架**：Song et al. (2021, Score SDE) 证明扩散模型和得分匹配模型可以被统一为随机微分方程（SDE）框架——前向过程是 SDE，反向过程是对应的逆时 SDE。
- **前向 SDE (Forward SDE)**：描述数据逐步演化为噪声的过程，由漂移系数 $f(x, t)$ 和扩散系数 $g(t)$ 定义：$dx = f(x, t)dt + g(t)dw$。
- **反向 SDE (Reverse SDE)**：描述从噪声恢复数据的过程，由漂移系数 $f(x, t) - g(t)^2 \nabla_x \log p_t(x)$ 和扩散系数 $g(t)$ 定义。
- **概率流 ODE (Probability Flow ODE)**：与 SDE 对应的确定性 ODE，其轨迹定义的边际分布 $p_t(x)$ 与 SDE 完全相同，但采样时可以走确定性的路径（加速采样）。
- **三种离散化的统一**：DDPM、得分匹配（NCSN）和 SDE 框架都被证明是同一个连续时间 SDE 的不同离散化方式——区别在于噪声调度和离散化策略。
- **时间相关的得分函数**：$\nabla_x \log p_t(x)$ 是连续时间下的得分函数，通过训练一个时间条件网络 $s_\theta(x, t)$ 来逼近，$t \in [0, T]$ 连续取值。

## 数学推导

**连续时间扩散过程的 SDE 形式**：

前向 SDE（Itô 形式）：

$$
dx = f(x, t) dt + g(t) dw
$$

其中 $f(\cdot, t): \mathbb{R}^d \to \mathbb{R}^d$ 是漂移系数，$g(t): \mathbb{R} \to \mathbb{R}$ 是扩散系数，$w$ 是标准 Wiener 过程（布朗运动）。

**DDPM 作为 SDE 的离散化**：

- 方差保持 SDE (VP-SDE)：$f(x, t) = -\frac{1}{2}\beta(t)x$, $g(t) = \sqrt{\beta(t)}$
- 对应的离散化就是 DDPM：$x_{t+\Delta t} = \sqrt{1-\beta(t)\Delta t} x_t + \sqrt{\beta(t)\Delta t} \epsilon$
- 当 $\Delta t \to 0$ 时收敛到 SDE

**反向 SDE（Anderson 定理）**：

$$
dx = [f(x, t) - g(t)^2 \nabla_x \log p_t(x)] dt + g(t) d\bar{w}
$$

其中 $\bar{w}$ 是逆时的 Wiener 过程，$dt$ 是负时间步。

关键点：反向 SDE 需要知道得分函数 $\nabla_x \log p_t(x)$——这正是需要学习的。

**概率流 ODE (Probability Flow ODE)**：

$$
dx = \left[f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x)\right] dt
$$

这个 ODE 与反向 SDE 产生相同的边际分布 $p_t(x)$，但：
- 确定性：给定初始 $x_T$，$x_0$ 完全确定
- 可以（在理论上）用 ODE 求解器（如 RK45）加速采样
- 可以计算数据的对数似然 $\log p(x)$（通过瞬时变量变换公式）

**VE-SDE 与 VP-SDE**：

- VP-SDE（Variance Preserving）：DDPM 类型，方差有界，$p_T \to \mathcal{N}(0, I)$
- VE-SDE（Variance Exploding）：NCSN 类型，方差无界，随 $t$ 增长趋向 $\mathcal{N}(0, \sigma_{\max}^2 I)$

## 直观理解

- **SDE 视角 = 从台阶到斜坡**：离散扩散模型（DDPM）像是一座楼梯——图像到噪声是 1000 级台阶。SDE 视角把台阶变成了一个平滑的斜坡——我们可以在斜坡上的任意位置计算速度和方向。
- **概率流 ODE = 在雾中沿着一条确定的路径走**：SDE 反向过程像是在雾中走路——每次迈步都有一点随机性。概率流 ODE 告诉我们"沿着最可能的路径走"——更快但更冒险。
- **为什么 SDE 视角重要**：它把离散的方法统一起来，就像牛顿和莱布尼茨把各种求面积的方法统一为微积分。有了统一的数学框架，可以推导出新的采样器（如 DPM-Solver 基于 ODE 的高阶求解器）。
- **VE vs VP**：VP-SDE 保持方差不变（像封闭系统中的能量守恒），VE-SDE 让方差不断增大（像把墨水滴入不断扩大的水箱）。两者最终等价，只是参数化不同。

## 代码示例

```python
import torch
import numpy as np
import math

class SDE:
    """SDE 基类"""
    def __init__(self, N=1000):
        self.N = N  # 离散化步数
    
    def sde(self, x, t):
        """前向 SDE: dx = f(x,t)dt + g(t)dw"""
        raise NotImplementedError
    
    def reverse_sde(self, x, t, score_fn):
        """反向 SDE: dx = [f - g^2 * score]dt + g dw"""
        f, g = self.sde(x, t)
        score = score_fn(x, t)
        drift = f - g**2 * score
        return drift, g
    
    def prob_flow_ode(self, x, t, score_fn):
        """概率流 ODE: dx = [f - 0.5 * g^2 * score]dt"""
        f, g = self.sde(x, t)
        score = score_fn(x, t)
        drift = f - 0.5 * g**2 * score
        return drift  # ODE 没有扩散项

class VPSDE(SDE):
    """方差保持 SDE (对应 DDPM)"""
    def __init__(self, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.beta_0 = beta_min
        self.beta_1 = beta_max
    
    def beta(self, t):
        """时间相关的噪声率"""
        return self.beta_0 + t * (self.beta_1 - self.beta_0)
    
    def sde(self, x, t):
        """VP-SDE: dx = -0.5 * beta(t) * x * dt + sqrt(beta(t)) * dw"""
        beta_t = self.beta(t)
        f = -0.5 * beta_t * x
        g = torch.sqrt(beta_t)
        return f, g
    
    def marginal_prob(self, x0, t):
        """计算 p_t(x|x_0) 的均值和标准差"""
        log_mean_coeff = -0.25 * t**2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        mean = torch.exp(log_mean_coeff) * x0
        std = torch.sqrt(1 - torch.exp(2 * log_mean_coeff))
        return mean, std

class VESDE(SDE):
    """方差膨胀 SDE (对应 NCSN)"""
    def __init__(self, sigma_min=0.01, sigma_max=50, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sde(self, x, t):
        """VE-SDE: dx = sqrt(d(sigma^2)/dt) * dw"""
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        g = torch.sqrt(2 * sigma * math.log(self.sigma_max / self.sigma_min) * sigma)
        f = -0. * x  # 零漂移
        return f, g
    
    def marginal_prob(self, x0, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x0
        std = sigma
        return mean, std

# 演示 SDE 采样
def euler_maruyama_sampling(sde, score_fn, n_samples=4, device='cpu', img_shape=(1, 3, 32, 32)):
    """使用 Euler-Maruyama 离散化求解反向 SDE"""
    # 从先验分布初始化
    x_T = torch.randn(n_samples, *img_shape).to(device)
    x_t = x_T
    
    dt = -1.0 / sde.N  # 负时间步
    
    for i in range(sde.N):
        t = torch.ones(n_samples, device=device) * (1 - i / sde.N)
        
        # 计算反向 SDE 的漂移和扩散系数
        drift, g = sde.reverse_sde(x_t, t[:, None, None, None], score_fn)
        
        # Euler-Maruyama 更新
        noise = torch.randn_like(x_t)
        x_t = x_t + drift * dt + g * noise * torch.sqrt(-dt)
    
    return x_t

def ode_sampling(sde, score_fn, n_samples=4):
    """使用概率流 ODE 采样（确定性）"""
    x_T = torch.randn(n_samples, 1, 3, 32, 32)
    x_t = x_T
    
    dt = -1.0 / sde.N
    
    for i in range(sde.N):
        t = torch.ones(n_samples) * (1 - i / sde.N)
        drift = sde.prob_flow_ode(x_t, t[:, None, None, None], score_fn)
        x_t = x_t + drift * dt
    
    return x_t

print("=== 扩散模型的 SDE 视角 ===")
print()
print("DDPM → VP-SDE (Variance Preserving)")
print("NCSN → VE-SDE (Variance Exploding)")
print()
print("统一框架:")
print("  前向 SDE:  dx = f(x,t)dt + g(t)dw")
print("  反向 SDE:  dx = [f - g^2 * s_theta(x,t)]dt + g dw̄")  
print("  概率流 ODE: dx = [f - 0.5 * g^2 * s_theta(x,t)]dt")
print()

# 验证 SDE 属性
vp = VPSDE(beta_min=0.1, beta_max=20)
ve = VESDE(sigma_min=0.01, sigma_max=50)

x0 = torch.randn(4, 3, 32, 32)
print("VP-SDE 边际分布 (t=0.5):")
mean, std = vp.marginal_prob(x0, torch.tensor(0.5))
print(f"  均值 norm: {mean.norm().item():.2f}, 标准差: {std.item():.2f}")

print("\nVE-SDE 边际分布 (t=0.5):")
mean, std = ve.marginal_prob(x0, torch.tensor(0.5))
print(f"  均值 norm: {mean.norm().item():.2f}, 标准差: {std.item():.2f}")
```

## 深度学习关联

- **DPM-Solver / DPM-Solver++**：利用概率流 ODE 的线性结构（半线性 ODE）设计专门的解析求解器，在 10-25 步内达到 1000 步 DDPM 的采样质量——这是 SDE 统一框架的直接工程收益。
- **Karras et al. (EDM)**：提出了一个精心设计的 SDE 框架（Euler Diffusion Model），通过系统性地设计噪声调度、损失权重和采样器，建立了当前扩散模型训练的"食谱"。
- **SDE 视角下的可控生成（SDEdit）**：反转 SDE 的特性使得可以在生成过程中的任意中间状态注入"部分噪声"——例如将用户涂鸦先用前向 SDE 加噪到某个 $t_0$，再用反向 SDE 去噪，实现基于引导的图像编辑。
- **Instant Flow / Rectified Flow**：将概率流 ODE 进一步简化为直线路径（从数据分布到噪声分布的直线流），使得一步生成成为可能——这是从 SDE 框架到 flow matching 范式的自然延伸。
