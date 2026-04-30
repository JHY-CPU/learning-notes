# 18 Nesterov Accelerated Gradient (NAG) 的前瞻性

## 核心概念

- **NAG 的定义**：Nesterov Accelerated Gradient（NAG）是 Momentum 的改进版本。它的核心创新是在计算梯度之前先"向前看一步"——在参数 $\theta_t$ 的基础上先加上动量项 $\beta v_t$，然后在这个"前瞻位置"上计算梯度。

- **Momentum 的问题**：标准动量在梯度方向即将改变时（比如接近最优点时），由于历史动量的惯性，容易"冲过头"（overshoot）。NAG 通过先估计下一步的位置然后再修正，有效减少了过冲。

- **前瞻性修正**：NAG 的过程可以理解为：先沿着动量方向预估 $\tilde{\theta} = \theta_t + \beta v_t$，然后在 $\tilde{\theta}$ 处计算梯度并修正方向。如果预估方向即将出错（如接近极值点），前瞻梯度可以提前刹车。

- **收敛速度**：在凸优化理论中，NAG 的收敛速度达到 $O(1/T^2)$，优于标准 Momentum 的 $O(1/T)$。在实践中，NAG 在深度神经网络训练中也通常比 Momentum 更快更稳定。

## 数学推导

**标准 Momentum**：

$$
v_{t+1} = \beta v_t + \eta \nabla L(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - v_{t+1}
$$

**Nesterov Accelerated Gradient**：

$$
\tilde{\theta}_t = \theta_t + \beta v_t
$$

$$
v_{t+1} = \beta v_t + \eta \nabla L(\tilde{\theta}_t)
$$

$$
\theta_{t+1} = \theta_t - v_{t+1}
$$

**等价形式**（更常见的实现写法）：

$$
v_{t+1} = \beta v_t + \eta \nabla L(\theta_t + \beta v_t)
$$

$$
\theta_{t+1} = \theta_t - v_{t+1}
$$

**展开对比**：

Momentum 更新：$\theta_{t+1} = \theta_t - \beta v_t - \eta \nabla L(\theta_t)$

NAG 更新：$\theta_{t+1} = \theta_t - \beta v_t - \eta \nabla L(\theta_t + \beta v_t)$

区别在于计算梯度的位置。NAG 在参数移动一步后的位置计算梯度，相当于"踩刹车"功能：如果前方即将上坡（梯度方向反转），前瞻梯度会提前减小更新量。

**理论收敛速度**：

对于凸 $L$-smooth 函数：

- SGD: $O(1/\sqrt{T})$（非平滑）或 $O(1/T)$（强凸）
- Momentum: $O(1/T)$
- NAG: $O(1/T^2)$

在二次型目标函数中，NAG 可以达到最优的收敛常数。

## 直观理解

NAG 的"前瞻性"可以用"开车"类比：Momentum 是"先看后视镜再加速"（根据当前位置的坡度决定下一步），而 NAG 是"先看前方路况再踩油门"（根据预料中的未来位置调整速度）。

具体来说，在开车下山时：
- Momentum：只看脚下坡度决定加速还是刹车，容易在坡底冲过头。
- NAG：先预估一下如果按当前速度滑行会到哪里（前方位置），然后看那个位置的坡度来决定是否刹车。如果前方即将上坡，就提前减速。

另一个直观理解是"打篮球上篮"：Momentum 是直接起跳（根据当前位置发力），NAG 是先跨出一步再起跳（根据预测位置调整发力）。后者能更好地根据篮筐距离调节力度，减少过冲。

NAG 的修正效果可以用一个简单场景演示：当梯度突然反向时（例如越过了极值点），Momentum 由于惯性继续向原方向移动，造成过冲和震荡；NAG 因为在新位置计算了梯度，能提前感知到方向变化并修正。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 手动实现 NAG 以展示其工作机制
def nag_manual(params, grads_func, lr=0.1, beta=0.9, steps=50):
    """手动实现 NAG — 展示前瞻机制"""
    v = [torch.zeros_like(p) for p in params]
    trajectory = [p.clone().detach() for p in params]

    for t in range(steps):
        # 1. 前瞻: 先沿着动量方向走一步
        lookahead = [p + beta * vi for p, vi in zip(params, v)]

        # 2. 在"前瞻位置"计算梯度
        lookahead_grads = grads_func(lookahead)

        # 3. 更新速度: 用前瞻位置的梯度
        for i in range(len(params)):
            v[i] = beta * v[i] + lr * lookahead_grads[i]

        # 4. 从原始位置更新（不是从前瞻位置）
        for i in range(len(params)):
            params[i] -= v[i]

        trajectory.append([p.clone().detach() for p in params])

    return trajectory

# 演示 NAG 在过冲控制上的优势
# 目标: 最小化 f(x) = x^2，从 x=5 开始
torch.manual_seed(42)

def test_optimizers():
    # SGD
    x_sgd = torch.tensor([5.0], requires_grad=True)
    opt_sgd = optim.SGD([x_sgd], lr=0.5)

    # Momentum
    x_mom = torch.tensor([5.0], requires_grad=True)
    opt_mom = optim.SGD([x_mom], lr=0.5, momentum=0.9)

    # NAG
    x_nag = torch.tensor([5.0], requires_grad=True)
    opt_nag = optim.SGD([x_nag], lr=0.5, momentum=0.9, nesterov=True)

    traj_sgd, traj_mom, traj_nag = [5.0], [5.0], [5.0]

    for step in range(30):
        # SGD
        opt_sgd.zero_grad()
        loss_sgd = x_sgd[0] ** 2
        loss_sgd.backward()
        opt_sgd.step()
        traj_sgd.append(x_sgd[0].item())

        # Momentum
        opt_mom.zero_grad()
        loss_mom = x_mom[0] ** 2
        loss_mom.backward()
        opt_mom.step()
        traj_mom.append(x_mom[0].item())

        # NAG
        opt_nag.zero_grad()
        loss_nag = x_nag[0] ** 2
        loss_nag.backward()
        opt_nag.step()
        traj_nag.append(x_nag[0].item())

    print(f"SGD 路径 (5 -> {traj_sgd[-1]:.4f}): 过冲={max(abs(v) for v in traj_sgd):.4f}")
    print(f"Momentum 路径 (5 -> {traj_mom[-1]:.4f}): 过冲={max(abs(v) for v in traj_mom):.4f}")
    print(f"NAG 路径 (5 -> {traj_nag[-1]:.4f}): 过冲={max(abs(v) for v in traj_nag):.4f}")

    return traj_sgd, traj_mom, traj_nag

traj_sgd, traj_mom, traj_nag = test_optimizers()

# 实际训练对比
print("\n在简单 MLP 上的训练对比:")
def train_with_optimizer(opt_name, opt_class, opt_kwargs):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(10, 64), nn.ReLU(), nn.Linear(64, 1))
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    criterion = nn.MSELoss()

    x = torch.randn(200, 10)
    y = torch.sin(x.sum(1, keepdim=True))

    for epoch in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()

print(f"SGD: {train_with_optimizer('SGD', optim.SGD, {'lr': 0.01}):.6f}")
print(f"Mom: {train_with_optimizer('Mom', optim.SGD, {'lr': 0.01, 'momentum': 0.9}):.6f}")
print(f"NAG: {train_with_optimizer('NAG', optim.SGD, {'lr': 0.01, 'momentum': 0.9, 'nesterov': True}):.6f}")
```

## 深度学习关联

- **PyTorch 中的 NAG**：在 PyTorch 中，NAG 是 `torch.optim.SGD` 的一个参数选项（`nesterov=True`）。使用 NAG 仅需将 `momentum` 参数设置为非零值并添加 `nesterov=True`，非常简单。

- **在视觉任务中的表现**：NAG 在计算机视觉任务中通常优于标准 Momentum，尤其是在需要精细收敛的任务（如目标检测、语义分割）中。经典的 ResNet 训练就使用了 NAG 优化器。

- **与 Adam 的关系**：Adam 优化器本身已经包含了一阶动量（momentum），相当于在自适应学习率的基础上叠加了 Momentum 的效果。也有一些工作尝试将 Nesterov 的前瞻思想引入 Adam（称为 Nadam 或 Nesterov Adam），在某些任务上取得了更好的效果。
