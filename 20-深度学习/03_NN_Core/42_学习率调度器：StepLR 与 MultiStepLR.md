# 42_学习率调度器：StepLR 与 MultiStepLR

## 核心概念

- **学习率调度的必要性**：学习率是训练神经网络最重要的超参数之一。固定学习率往往效果不佳——训练初期需要大学习率快速收敛，后期需要小学习率精细微调。学习率调度器（Scheduler）在训练过程中动态调整学习率。
- **StepLR 定义**：StepLR 每隔固定步数将学习率乘以一个衰减因子 $\gamma$：$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$，其中 $T$ 是步长（step_size），$\gamma$ 是衰减率（通常 0.1）。
- **MultiStepLR 定义**：MultiStepLR 在预定义的里程碑（milestones）时刻衰减学习率，而不是每隔固定步数。例如 milestones=[30, 60, 80] 表示在第 30、60、80 个 epoch 时学习率乘以 $\gamma$。
- **两者的对比**：StepLR 简单均匀地衰减，适合对训练过程没有深入了解的情况。MultiStepLR 更加灵活，可以在关键阶段（如损失 plateau 时）手动触发衰减，通常效果更好。

## 数学推导

**StepLR**：

$$
\eta_{epoch} = \eta_0 \cdot \gamma^{\lfloor epoch / step\_size \rfloor}
$$

其中 $\gamma \in (0,1)$ 是衰减因子，$step\_size$ 是衰减间隔。

例如：$\eta_0 = 0.1$，$\gamma = 0.1$，$step\_size = 30$：
- epoch 0-29: $\eta = 0.1$
- epoch 30-59: $\eta = 0.01$
- epoch 60-89: $\eta = 0.001$

**MultiStepLR**：

$$
\eta_{epoch} = \eta_0 \cdot \gamma^{c(epoch)}
$$

其中 $c(epoch) = \sum_{m \in milestones} \mathbb{I}(epoch \geq m)$，即当前已经经过的里程碑数量。

例如：$\eta_0 = 0.1$，$\gamma = 0.1$，$milestones = [30, 60, 80]$：
- epoch 0-29: $\eta = 0.1$
- epoch 30-59: $\eta = 0.01$
- epoch 60-79: $\eta = 0.001$
- epoch 80+: $\eta = 0.0001$

**学习率衰减的理论动机**：

从 SGD 收敛性理论看，学习率需要满足：
- $\sum_{t=1}^\infty \eta_t = \infty$（保证能到达任何区域）
- $\sum_{t=1}^\infty \eta_t^2 < \infty$（保证收敛）

分段衰减的 $\eta_t$ 满足这些条件。直觉上，训练初期远离最优点，需要大步长；后期接近最优点，需要小步长来精细搜索。

## 直观理解

学习率调度就像"开车策略"：
- 训练初期（高速公路）：大学习率，快速前进
- 训练中期（普通公路）：中等学习率，稳定前进  
- 训练后期（小巷子）：小学习率，精细调整

StepLR 是"定时减速"——每开 30 分钟减速一次（速度减半）。MultiStepLR 是"看到路况变化才减速"——前面有弯道（损失的 plateau）时才减速。

MultiStepLR 更灵活的原因在于，训练过程中损失的下降并非均匀的。通常在训练的前期和某些关键节点（如模型开始拟合复杂模式时），损失下降较快。在这些节点之后，需要降低学习率以进一步精细化。MultiStepLR 允许研究者根据经验设定这些关键节点。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

# StepLR 和 MultiStepLR 演示
torch.manual_seed(42)

model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# StepLR: 每 30 个 epoch 衰减 0.1 倍
scheduler_step = scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# MultiStepLR: 在 epoch 30, 60, 80 衰减
scheduler_multi = scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)

print("学习率调度对比:")
print(f"{'Epoch':<10} {'StepLR':<15} {'MultiStepLR':<15}")
print("-" * 40)
for epoch in range(100):
    scheduler_step.step()
    scheduler_multi.step()
    if epoch % 10 == 0 or epoch in [30, 60, 80]:
        lr_step = scheduler_step.get_last_lr()[0]
        lr_multi = scheduler_multi.get_last_lr()[0]
        print(f"{epoch:<10} {lr_step:<15.6f} {lr_multi:<15.6f}")

# 分布式训练示例
print("\n训练 CIFAR 风格分类器:")
def train_with_scheduler(scheduler_class, scheduler_kwargs, name):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64*8*8, 10)
    )
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    scheduler = scheduler_class(opt, **scheduler_kwargs)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(200, 3, 32, 32)
    y = torch.randint(0, 10, (200,))

    for epoch in range(70):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()

    return loss.item()

step_lr_loss = train_with_scheduler(
    scheduler.StepLR, {"step_size": 30, "gamma": 0.1}, "StepLR")
multi_lr_loss = train_with_scheduler(
    scheduler.MultiStepLR, {"milestones": [30, 50], "gamma": 0.1}, "MultiStepLR")

print(f"  StepLR final loss: {step_lr_loss:.4f}")
print(f"  MultiStepLR final loss: {multi_lr_loss:.4f}")

# 与固定学习率的对比
def train_fixed_lr(lr):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(64*8*8, 10)
    )
    opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(200, 3, 32, 32)
    y = torch.randint(0, 10, (200,))

    for epoch in range(70):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

print("\n固定学习率对比:")
for lr in [0.001, 0.01, 0.1]:
    loss = train_fixed_lr(lr)
    print(f"  lr={lr:.3f}: loss={loss:.4f}")

# ExponentialLR（指数衰减）
print("\n其他常见学习率调度器:")
model = nn.Linear(10, 2)
opt = optim.SGD(model.parameters(), lr=0.1)

exp_scheduler = scheduler.ExponentialLR(opt, gamma=0.95)
for epoch in range(10):
    exp_scheduler.step()
    print(f"  ExponentialLR epoch {epoch}: lr={exp_scheduler.get_last_lr()[0]:.6f}")
```

## 深度学习关联

- **传统视觉任务的标准配置**：StepLR 和 MultiStepLR 是训练标准 CNN 模型（如 ResNet、VGG）的传统选择。例如，ResNet-50 在 ImageNet 上的标准训练使用 MultiStepLR，milestones=[30, 60, 80]，$\gamma=0.1$，总训练 90 个 epoch。
- **被更高级调度器取代的趋势**：StepLR 和 MultiStepLR 正在被 Cosine Annealing 和 One Cycle Policy 等更先进的调度器取代，因为后者不需要手动指定衰减时刻，且通常效果更好。但 MultiStepLR 仍然是许多竞赛方案的首选，因为其简单、可控。
- **调度器与优化器的匹配**：不同的调度器适合不同的优化器。StepLR/MultiStepLR 与 SGD+Momentum 配合良好，因为 SGD 对大学习率变化不敏感。Adam 等自适应优化器通常配合 Cosine Annealing 或 Warmup 使用，因为其对学习率变化更敏感。
