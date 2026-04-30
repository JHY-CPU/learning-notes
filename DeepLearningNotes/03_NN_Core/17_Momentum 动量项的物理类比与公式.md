# 17 Momentum 动量项的物理类比与公式

## 核心概念

- **动量机制**：动量（Momentum）在 SGD 中引入了一个"速度"项，累积过去梯度的指数衰减平均。参数更新不仅依赖当前梯度，还受到历史梯度方向的影响，就像物理世界中物体的运动具有惯性。

- **物理类比**：动量的名称来自于物理学中的动量 $p = mv$。在优化中，参数就像在一个有摩擦力的曲面上下滑的小球——它沿着梯度的方向加速，但受到摩擦力的阻尼作用，不会无限加速。

- **超参数 $\beta$**：动量系数 $\beta$（通常取 0.9 或 0.99）控制历史梯度的衰减速度。$\beta$ 越大，历史梯度的影响越持久，运动越"平滑"；$\beta$ 越小，历史梯度的影响越短暂，运动越接近 SGD。

- **解决的两类问题**：动量解决了 SGD 的两大痛点——（1）在梯度方向变化剧烈的区域（如狭窄山谷中的"之字形"路径），动量可以平滑掉振荡；（2）在梯度较小的平坦区域，动量可以利用历史梯度积累的动能冲出平台。

## 数学推导

**标准 SGD**：

$$
\theta_{t+1} = \theta_t - \eta g_t
$$

其中 $g_t = \nabla L(\theta_t)$。

**SGD with Momentum**：

$$
v_{t+1} = \beta v_t + g_t
$$

$$
\theta_{t+1} = \theta_t - \eta v_{t+1}
$$

其中 $v_t$ 是速度（velocity），$\beta \in [0, 1)$ 是动量衰减系数。

**展开形式**：

$$
v_{t+1} = \beta v_t + g_t = g_t + \beta g_{t-1} + \beta^2 g_{t-2} + \cdots + \beta^t g_0
$$

这相当于对所有历史梯度进行指数加权移动平均（EWMA），权重呈指数衰减。因为 $\beta < 1$，越早的梯度权重越小。

**有效学习率分析**：

假设梯度 $g$ 恒定，则动量项的稳态值：

$$
v_{\infty} = \frac{g}{1 - \beta}
$$

这意味着动量的有效学习率为 $\eta / (1 - \beta)$。例如 $\beta = 0.9$ 时，有效学习率是名义学习率的 10 倍。

**振荡抑制分析**：

考虑梯度在垂直方向 $g_y$ 上交替变换方向（振荡），在水平方向 $g_x$ 上保持一致（目标方向）。动量对垂直分量进行抵消，对水平分量进行累加：

垂直方向：$v_y \approx \beta v_y - g_y \approx 0$（正负抵消）
水平方向：$v_x \approx \beta v_x + g_x \rightarrow g_x/(1-\beta)$（累加放大）

## 直观理解

动量的物理类比是最直观的：想象一个小球从山坡上滚下。如果没有动量（SGD），小球每次只根据当前位置的坡度决定移动方向——如果山谷很窄，小球会左右震荡，前进缓慢。有了动量，小球积累了"惯性"，即使遇到左右交替的坡度，它的主要运动方向仍然向前，只在垂直方向上有小幅振荡。

超参数 $\beta$ 相当于"摩擦力"的倒数。$\beta$ 越大（如 0.99），摩擦力越小，小球可以滑得更远，甚至冲上对面小坡（逃逸局部极小值）。$\beta$ 越小（如 0.5），摩擦力越大，小球运动更稳健、更保守。

动量对梯度噪声的平滑效果类似于"信号处理中的低通滤波器"——高频振荡（梯度噪声）被滤除，低频信号（真实梯度方向）被保留。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 手动实现 Momentum vs PyTorch 内置优化器
def sgd_momentum_manual(params, grads, velocities, lr, beta):
    """手动实现 SGD with Momentum"""
    with torch.no_grad():
        for i, (p, g) in enumerate(zip(params, grads)):
            velocities[i] = beta * velocities[i] + g
            p -= lr * velocities[i]

# 比较 SGD 和 Momentum 在振荡情况下的表现
torch.manual_seed(42)

# 构造一个具有"狭窄山谷"特征的损失函数
# f(x,y) = 0.1*x^2 + 10*y^2 (x 方向平缓，y 方向陡峭)
def loss_fn(x, y):
    return 0.1 * x**2 + 10 * y**2

def optimize(optimizer_class, opt_kwargs, n_steps=50):
    x = torch.tensor([5.0, 1.0], requires_grad=True)  # 起点
    optimizer = optimizer_class([x], **opt_kwargs)
    trajectory = [x.detach().clone().numpy().tolist()]

    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = loss_fn(x[0], x[1])
        loss.backward()
        optimizer.step()
        trajectory.append(x.detach().clone().numpy().tolist())

    return trajectory

# SGD 无动量
traj_sgd = optimize(optim.SGD, {"lr": 0.1})
# SGD + Momentum
traj_mom = optimize(optim.SGD, {"lr": 0.1, "momentum": 0.9})

print(f"SGD 终点: x={traj_sgd[-1][0]:.4f}, y={traj_sgd[-1][1]:.4f}, "
      f"loss={loss_fn(traj_sgd[-1][0], traj_sgd[-1][1]):.6f}")
print(f"Momentum 终点: x={traj_mom[-1][0]:.4f}, y={traj_mom[-1][1]:.4f}, "
      f"loss={loss_fn(traj_mom[-1][0], traj_mom[-1][1]):.6f}")

# 观察路径振荡
print(f"\nSGD y 方向振荡: {max(traj_sgd, key=lambda p: abs(p[1]))[1]:.4f}")
print(f"Momentum y 方向振荡: {max(traj_mom, key=lambda p: abs(p[1]))[1]:.4f}")

# 实际训练示例: 简单 CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 32, 3)
        self.fc = nn.Linear(32*26*26, 10)

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x.view(x.size(0), -1))

def test_optimizer(name, optimizer_class, opt_kwargs):
    model = SimpleCNN()
    optimizer = optimizer_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    # 模拟数据
    x = torch.randn(64, 1, 28, 28)
    y = torch.randint(0, 10, (64,))

    losses = []
    for epoch in range(50):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses[-1]

print(f"\nSGD 最终损失: {test_optimizer('SGD', optim.SGD, {'lr': 0.01}):.4f}")
print(f"Momentum 最终损失: {test_optimizer('Momentum', optim.SGD, {'lr': 0.01, 'momentum': 0.9}):.4f}")
```

## 深度学习关联

- **几乎所有优化器的标配**：动量项是现代深度学习优化器的标配。SGD + Momentum 是计算机视觉领域（ResNet、YOLO 等）最常用的优化配置。即使 Adam 也内置了动量机制（通过 $\beta_1$ 参数）。

- **逃逸局部极小值和平坦区域**：动量可以帮助优化过程逃逸局部极小值和梯度接近零的平坦区域。在平坦区域，SGD 的梯度很小导致停滞，但动量积累的历史梯度可以提供"冲量"穿越平台。

- **Nesterov 动量的改进**：Nesterov Accelerated Gradient（NAG）是 Momentum 的改进版本，它在计算梯度时使用"前瞻"位置 $\theta_t + \beta v_t$ 而不是当前位置。这种修正使得参数更新更加稳定，在某些场景下收敛更快。
