# 20 RMSProp：解决 AdaGrad 学习率单调递减

## 核心概念

- **RMSProp 的核心改进**：RMSProp（Root Mean Square Propagation）解决了 AdaGrad 学习率单调递减的致命缺陷。关键改进是将梯度平方的累积求和替换为指数加权移动平均（EWMA），使得过去梯度的权重指数衰减，而非线性累积。

- **衰减系数 $\beta$**：$\beta$（通常取 0.9）控制历史梯度的衰减速度。RMSProp 维护一个运行均值 $v_t = \beta v_{t-1} + (1-\beta)g_t^2$，$v_t$ 代表当前梯度的 RMS（均方根）。与 AdaGrad 的 $v_t$ 不同，RMSProp 的 $v_t$ 不会无限增长。

- **非单调学习率**：由于 $v_t$ 是移动均值而非单调递增的累积和，RMSProp 的学习率可以上升也可以下降。若当前梯度很小，$v_t$ 减小，学习率增大；反之亦然。这种自适应特性使 RMSProp 能适应非平稳目标函数。

- **逐参数自适应**：每个参数拥有独立的自适应学习率，基于该参数的梯度历史调整。梯度振荡剧烈的参数学习率降低，梯度稳定的参数学习率提高。

## 数学推导

**AdaGrad 的问题**：

$$
G_t = \sum_{\tau=1}^t g_\tau^2, \quad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t + \epsilon}} \cdot g_t
$$

$G_t$ 单调递增到无穷大 $\Rightarrow$ 学习率 $\to 0$。

**RMSProp 更新**：

$$
v_t = \beta v_{t-1} + (1-\beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t + \epsilon}} \cdot g_t
$$

其中 $v_t$ 是梯度平方的指数加权移动平均，$\beta$ 是衰减系数（通常 0.9），$\epsilon$ 是数值稳定常数（通常 $10^{-8}$）。

**指数移动平均的特性**：

展开 $v_t$：

$$
v_t = (1-\beta)\sum_{\tau=1}^t \beta^{t-\tau} g_\tau^2
$$

有效窗口大小为 $1/(1-\beta)$：
- $\beta = 0.9$：窗口 $\approx 10$ 步
- $\beta = 0.99$：窗口 $\approx 100$ 步
- $\beta = 0.999$：窗口 $\approx 1000$ 步

**RMSProp 的另一种解释**：

更新量可以写为：

$$
\Delta\theta_t = -\eta \cdot \frac{g_t}{\sqrt{v_t + \epsilon}}
$$

分母 $\sqrt{v_t}$ 是对梯度尺度的估计。当梯度大时，分母大，缩小步长；当梯度小时，分母小，放大步长。这相当于对梯度进行"白化"——将不同尺度的梯度标准化到统一的量级。

**与 AdaGrad 的关键区别**：

AdaGrad: $v_t = v_{t-1} + g_t^2$（数值不断累积 $\uparrow$）
RMSProp: $v_t = \beta v_{t-1} + (1-\beta)g_t^2$（移动平均，可增可减）

## 直观理解

RMSProp 解决了 AdaGrad"学不动"的问题，其工作原理类似于"自适应音量调节"：

想象你在听收音机，信号有时强有时弱（梯度大小变化）。AdaGrad 的做法是记录从开机到现在的所有信号强度——信号越听越多，音量越调越低，最终什么都听不到。RMSProp 的做法是只关注最近一段时间的信号——信号强时降低音量，信号弱时提高音量，始终保持在可听水平。

这个"自适应音量"的调节机制在非平稳目标中特别重要。比如在训练 GAN 或强化学习时，目标函数本身不断变化。RMSProp 能迅速适应新的梯度尺度，而 AdaGrad 会被历史累积拖累。

超参数 $\beta$ 控制"记忆长度"：$\beta = 0.9$ 意味着主要关注最近 10 步的梯度信息，$\beta = 0.99$ 关注最近 100 步。更大的 $\beta$ 使学习率变化更平滑，但响应更慢。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 手动实现 RMSProp
def rmsprop_manual(params, grads, v, lr=0.01, beta=0.9, eps=1e-8):
    """手动 RMSProp"""
    with torch.no_grad():
        for i, (p, g) in enumerate(zip(params, grads)):
            v[i] = beta * v[i] + (1 - beta) * g ** 2
            p -= lr / (v[i] + eps).sqrt() * g

# 对比 AdaGrad 和 RMSProp 在非凸函数上的表现
torch.manual_seed(42)

# 使用波动剧烈的损失曲面: f(x) = x^2 + 2*sin(5*x)
def loss_fn(x):
    return x ** 2 + 2 * torch.sin(5 * x)

# 从 x=3 开始优化
def optimize_comparison(name, opt_class, opt_kwargs, steps=100):
    x = torch.tensor([3.0], requires_grad=True)
    optimizer = opt_class([x], **opt_kwargs)
    trajectory = [x.item()]

    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(x)
        loss.backward()
        optimizer.step()
        trajectory.append(x.item())

    final_loss = loss_fn(torch.tensor(trajectory[-1]))
    print(f"{name}: x_final={trajectory[-1]:.4f}, loss={final_loss.item():.4f}")
    return trajectory

print("AdaGrad vs RMSProp 在波动曲面上的表现:")
traj_adagrad = optimize_comparison("AdaGrad", optim.Adagrad, {"lr": 0.5}, 100)
traj_rmsprop = optimize_comparison("RMSProp", optim.RMSprop, {"lr": 0.01}, 100)

# 对比梯度平方累积 vs 移动平均
print("\n梯度平方的动态变化:")
v_adagrad = 0.0  # AdaGrad 累积
v_rmsprop = 0.0  # RMSProp 移动平均
beta = 0.9

for step in range(20):
    g = torch.randn(1).item() ** 2  # 模拟随机梯度平方
    v_adagrad += g
    v_rmsprop = beta * v_rmsprop + (1 - beta) * g
    if step % 5 == 0:
        print(f"  Step {step:2d}: AdaGrad v={v_adagrad:.4f}, RMSProp v={v_rmsprop:.4f}")

# 真实训练示例: 非平稳目标
print("\n训练 MLP 对比:")
def train_comparison(name, opt_class, opt_kwargs):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 1))
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    criterion = nn.MSELoss()

    x = torch.randn(500, 20)
    y = torch.sin(x.sum(1, keepdim=True))

    for epoch in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

print(f"SGD: {train_comparison('SGD', optim.SGD, {'lr': 0.01}):.6f}")
print(f"AdaGrad: {train_comparison('AdaGrad', optim.Adagrad, {'lr': 0.01}):.6f}")
print(f"RMSProp: {train_comparison('RMSProp', optim.RMSprop, {'lr': 0.001}):.6f}")
```

## 深度学习关联

- **RNN 训练的首选**：RMSProp 在循环神经网络（RNN）训练中表现优异，因为 RNN 的梯度具有高度非平稳性（随时间步变化剧烈）。RMSProp 的自适应学习率调节机制能有效应对 RNN 训练中的梯度尺度变化。

- **GAN 训练的核心优化器**：RMSProp 是训练生成对抗网络（GAN）的常用优化器。GAN 的训练涉及生成器和判别器的动态博弈，损失曲面极其非平稳，RMSProp 的自适应特性非常适合这种场景。

- **Adam 的前身**：RMSProp 为 Adam 优化器提供了二阶矩估计的核心思想。Adam 在 RMSProp 的基础上增加了一阶动量（对应 Momentum）和偏差校正，形成了目前最广泛使用的深度学习优化器。理解 RMSProp 是理解 Adam 的关键前提。
