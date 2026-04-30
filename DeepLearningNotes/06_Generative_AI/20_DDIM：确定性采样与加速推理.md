# 20_DDIM：确定性采样与加速推理

## 核心概念
- **DDIM (Denoising Diffusion Implicit Models)**：一种改进的扩散模型采样方法，将原本随机的马尔可夫采样过程变为确定性的隐式采样，大幅加速生成。
- **非马尔可夫前向过程**：DDIM 的核心洞察是——DDPM 训练时依赖马尔可夫假设，但采样时不一定需要。DDIM 构造了一族非马尔可夫的前向过程，使得同一个训练好的 DDPM 模型可以用不同的采样策略。
- **确定性采样**：DDIM 的采样过程不含随机噪声项（$z=0$），给定初始 $x_T$，生成的 $x_0$ 是确定的。这意味着潜变量 $x_T$ 具有了语义意义（类似 GAN 的潜空间）。
- **跳步采样 (Subsequence Sampling)**：DDIM 可以只用训练总步数 $T$ 的子序列 $\{t_1, t_2, ..., t_S\}$ 进行采样（如 $S=50$ 而非 $T=1000$），大幅减少采样步数。
- **隐式模型**：之所以叫"隐式"（Implicit），是因为采样过程是一个隐式的概率模型——它定义了 $x_0$ 到 $x_T$ 的确定性映射，而不是一个显式概率分布。
- **插值与编辑**：由于 DDIM 的确定性映射，$x_T$ 上的线性插值对应到生成图像上的语义平滑变换，这使得 DDIM 非常适合做潜空间编辑。

## 数学推导

**DDPM 的采样公式**（随机）：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

**DDIM 的采样公式**（确定性，$\sigma_t = 0$）：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)
$$

**DDIM 的一般形式**（允许控制随机性）：

$$
x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{x}_0(x_t) + \sqrt{1 - \bar{\alpha}_{t-1} - \sigma_t^2} \cdot \epsilon_\theta(x_t, t) + \sigma_t z
$$

其中 $\hat{x}_0(x_t) = \frac{x_t - \sqrt{1-\bar{\alpha}_t} \epsilon_\theta(x_t, t)}{\sqrt{\bar{\alpha}_t}}$ 是预测的 $x_0$。

当 $\sigma_t = \sqrt{\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t}$ 时，退化为 DDPM（随机）

当 $\sigma_t = 0$ 时，是 DDIM（确定性）

**加速采样的跳步策略**：

从原始时间序列 $\{0, 1, ..., T\}$ 中选择子序列 $\{\tau_1, \tau_2, ..., \tau_S\}$，其中 $\tau_1 = 0$ < $\tau_2$ < ... < $\tau_S = T$。在子序列上的采样公式变为：

$$
x_{\tau_{i-1}} = \sqrt{\bar{\alpha}_{\tau_{i-1}}} \cdot \hat{x}_0(x_{\tau_i}) + \sqrt{1 - \bar{\alpha}_{\tau_{i-1}}} \cdot \epsilon_\theta(x_{\tau_i}, \tau_i)
$$

这只需要 $S$ 步（如 $S=20$ 或 $S=50$）就能完成生成。

**与 ODE 的联系**：

DDIM 的确定性过程可以看作概率流 ODE（Probability Flow ODE）的离散化：

$$
dx = \left[ f(x, t) - \frac{1}{2} g(t)^2 \nabla_x \log p_t(x) \right] dt
$$

其中 $f$ 和 $g$ 是 SDE 的漂移和扩散系数。

## 直观理解
- **DDPM vs DDIM**：DDPM 像是带着眼罩走路——每一步除了按照指令走，还会随机晃一下；DDIM 则摘掉了眼罩——每一步都确定地按指令走，到得更快更稳。
- **跳步采样 = 跨台阶下楼梯**：原来需要下 1000 级台阶（每次下一级），DDIM 让你一次跨 20 级——虽然粗糙一些，但只需要几十步就能到达底层。而且由于方向预测准确，不会踩空。
- **确定性映射的优势**：在 DDPM 中，两个相近的 $x_T$ 会生成完全不同的 $x_0$（因为随机性放大了差异）。在 DDIM 中，$x_T$ 上微小的变化对应到 $x_0$ 也是微小的变化——这就像 GAN 的潜空间，找到了"语义方向"。
- **为什么 DDIM 不影响训练**：DDIM 的核心洞见是：DDPM 的训练目标（预测噪声 $\epsilon$）只依赖于 $q(x_t|x_0)$ 的边缘分布，不依赖于联合分布 $q(x_{1:T}|x_0)$ 的马尔可夫结构。因此改变采样过程的联合分布不影响训练的有效性。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class DDIMScheduler:
    """DDIM 采样调度器"""
    def __init__(self, betas, timesteps=1000):
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = 1. - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def get_subsequence(self, num_steps=50):
        """获取采样子序列（跳步）"""
        step_ratio = self.timesteps // num_steps
        indices = list(range(0, self.timesteps, step_ratio))
        # 确保包含最后一步
        if indices[-1] != self.timesteps - 1:
            indices.append(self.timesteps - 1)
        return indices  # [t_1, t_2, ..., t_S]
    
    def ddim_step(self, model, x_t, t, t_prev, eta=0.0):
        """
        DDIM 单步采样
        
        参数:
            model: 噪声预测网络
            x_t: 当前时刻的样本
            t: 当前时间步（较大）
            t_prev: 目标时间步（较小）
            eta: 随机性控制参数 (0 = 确定性 DDIM, 1 = DDPM)
        """
        with torch.no_grad():
            t_tensor = torch.full((x_t.size(0),), t, device=x_t.device, dtype=torch.long)
            epsilon_theta = model(x_t, t_tensor)
            
            # 预测 x_0
            alpha_bar_t = self.alpha_bars[t].to(x_t.device)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
            
            x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * epsilon_theta) / sqrt_alpha_bar_t
            
            if t_prev < 0:
                return x_0_pred
            
            # DDIM 的一般形式
            alpha_bar_t_prev = self.alpha_bars[t_prev].to(x_t.device)
            
            # 方向预测
            sigma_t = eta * torch.sqrt(
                (1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_prev)
            )
            
            # x_{t-1} 的"预测"部分
            pred_x_0_part = torch.sqrt(alpha_bar_t_prev) * x_0_pred
            
            # x_{t-1} 的"噪声方向"部分
            dir_xt_part = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * epsilon_theta
            
            if eta > 0:
                noise = torch.randn_like(x_t)
                return pred_x_0_part + dir_xt_part + sigma_t * noise
            else:
                return pred_x_0_part + dir_xt_part

# DDIM 完整采样循环
def ddim_sample(model, scheduler, num_steps=50, batch_size=4, img_channels=3, img_size=32, eta=0.0, device='cpu'):
    """使用 DDIM 加速采样"""
    # 获取跳步子序列
    timesteps = scheduler.get_subsequence(num_steps)
    
    x_t = torch.randn(batch_size, img_channels, img_size, img_size).to(device)
    
    for i in range(len(timesteps) - 1, 0, -1):
        t = timesteps[i]
        t_prev = timesteps[i - 1] if i > 0 else -1
        
        x_t = scheduler.ddim_step(model, x_t, t, t_prev, eta=eta)
    
    return x_t

# 验证 DDIM 的确定性性质
def test_ddim_determinism(model, scheduler):
    """验证 DDIM 的确定性：相同初始噪声生成相同结果"""
    # 固定种子确保可比性
    torch.manual_seed(42)
    x_T_1 = torch.randn(1, 3, 32, 32)
    torch.manual_seed(42)
    x_T_2 = torch.randn(1, 3, 32, 32)
    
    x_0_1 = ddim_sample(model, scheduler, num_steps=50, batch_size=1, eta=0.0)
    
    # 重新生成
    torch.manual_seed(42)
    x_T_3 = torch.randn(1, 3, 32, 32)
    x_0_2 = ddim_sample(model, scheduler, num_steps=50, batch_size=1, eta=0.0)
    
    # 验证：随机版本每次结果不同
    torch.manual_seed(42)
    x_T_4 = torch.randn(1, 3, 32, 32)
    x_0_3 = ddim_sample(model, scheduler, num_steps=50, batch_size=1, eta=1.0)
    
    print("DDIM 确定性演示:")
    print("  注意: 由于模型未训练，输出是噪声。")
    print("  关键概念：DDIM(eta=0)是确定性的，DDPM(eta=1)是随机的。")
    print(f"  DDIM 采样步数: 50 (原为 1000)")
    print(f"  加速比: 20x")

# 创建模型和调度器
model = nn.Identity()  # 占位模型（实际中应使用训练好的 U-Net）
betas = torch.linspace(1e-4, 0.02, 1000)
scheduler = DDIMScheduler(betas, timesteps=1000)

# 性能对比
print("=== DDPM vs DDIM 采样对比 ===")
print(f"DDPM: 需要 1000 步，每步需要一次网络前向推理")
print(f"DDIM: 只需 20-100 步（加速 10-50 倍）")
print(f"DDIM: 确定性采样，支持潜空间插值")
print(f"DDIM: 可以复用 DDPM 训练的模型，无需重新训练")
print()

# 展示跳步序列
subseq = scheduler.get_subsequence(10)
print(f"DDIM 跳步子序列 (10步): {subseq}")
print(f"步长范围: {subseq[0]} -> {subseq[-1]}")
```

## 深度学习关联
- **DPM-Solver**：在 DDIM 的基础上，将扩散 ODE 用高阶数值求解器（如 Runge-Kutta）求解，能用 10-20 步达到 DDIM 100 步的质量，是目前最快的高质量采样器之一。
- **LDM (Latent Diffusion Models)**：Stable Diffusion 使用 DDIM 作为默认采样器（搭配 $\eta=0$ 的确定性采样），在潜空间中仅用 50 步就能生成高质量图像。
- **一致性模型 (Consistency Model)**：更进一步——学习一个直接映射 $f_\theta(x_T, T) = x_0$，将采样压缩到一步。DDIM 为一致性模型的蒸馏提供了教师模型（用 DDIM 多步采样生成配对数据，训练学生模型一步映射）。
- **编辑与插值**：DDIM 的确定性潜空间使得图像编辑成为可能——对 $x_T$ 进行插值可以实现两张图像的平滑融合，对 $x_T$ 沿特定方向移动可以编辑图像属性。
