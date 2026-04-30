# 44 One Cycle Learning Rate Policy

## 核心概念

- **One Cycle Policy 定义**：One Cycle 策略由 Leslie N. Smith 提出，在一个完整的训练周期内，先让学习率从较低值线性增加到较高值（预热阶段），再线性减少到较低值（退火阶段）。整个过程形成一个"三角形"或"倒 V 形"。

- **动量反向**：与学习率变化同步，动量（momentum）从高值降低到低值（在预热期），然后从低值回升到高值（在退火期）。这种"学习率上升、动量下降"的相反变化增强了训练的稳定性。

- **超收敛（Super-convergence）**：One Cycle 策略可以使训练收敛速度大大加快，达到"超收敛"效果。使用合适的 One Cycle 策略，模型可以在原本 1/5 到 1/10 的训练轮数内达到与标准训练相当的精度。

- **三个阶段**：完整的 One Cycle 包含三个阶段：
  1. **预热阶段**（~20-30% 总轮数）：学习率从 $\eta_{\min}$ 线性增加到 $\eta_{\max}$，动量从高到低
  2. **退火阶段**（~70-80% 总轮数）：学习率从 $\eta_{\max}$ 线性减少到 $\eta_{\min}$
  3. **精细调整阶段**（可选）：最后一部分学习率继续衰减到更低值

## 数学推导

**One Cycle 学习率调度**：

预热阶段（$0 < t < T_{\text{warmup}}$）：

$$
\eta_t = \eta_{\min} + \frac{t}{T_{\text{warmup}}}(\eta_{\max} - \eta_{\min})
$$

退火阶段（$T_{\text{warmup}} \leq t < T$）：

$$
\eta_t = \eta_{\max} - \frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}}(\eta_{\max} - \eta_{\min})
$$

**动量调度**（与学习率反向）：

预热阶段：

$$
momentum_t = momentum_{\max} - \frac{t}{T_{\text{warmup}}}(momentum_{\max} - momentum_{\min})
$$

退火阶段：

$$
momentum_t = momentum_{\min} + \frac{t - T_{\text{warmup}}}{T - T_{\text{warmup}}}(momentum_{\max} - momentum_{\min})
$$

典型参数：$\eta_{\min} = 10^{-5}$，$\eta_{\max} = 10^{-2} \sim 10^{-1}$，$momentum_{\max} = 0.95$，$momentum_{\min} = 0.85$。

**超收敛的理论解释**：

1. **预热阶段**：低学习率允许网络在初始阶段稳定训练，避免早期梯度爆炸。同时高动量帮助网络保持方向一致。

2. **高学习率阶段**：学习率达到峰值。高学习率在非凸优化中有正则化效果，帮助网络逃离尖锐的局部极小值，找到更平坦的解。

3. **退火阶段**：学习率降低到小值，进行精细收敛。同时动量回升到高值，提供收敛的稳定性。

**最大学习率的确定**：

Smith 提出了"LR range test"来确定 $\eta_{\max}$：在几个 mini-batch 上线性增加学习率，观察损失的变化，找到损失开始发散前的最大可接受学习率。

## 直观理解

One Cycle 策略可以类比为"跳高"——助跑（预热阶段）逐步加速，起跳（高学习率阶段）达到最高点，然后下落（退火阶段）平稳着地。

另一个类比是"探测-利用"过程：
- **预热**（探测准备）：以保守的步伐开始，积累动量
- **高学习率**（大范围探测）：大胆探索参数空间，找到有希望的区域
- **退火**（精细利用）：缩小步长，在最佳区域精细调整

学习率和动量的反向变化非常巧妙：训练初期，大动量 + 小学习率 = 稳定探索；训练中期，小动量 + 大学习率 = 大胆探索；训练后期，大动量 + 小学习率 = 稳定收敛。

超收敛效果的直觉：传统训练通常在低学习率下花费大量时间，One Cycle 通过在高学习率阶段快速穿越参数空间，在更少的步数内到达相似的最终位置。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

# One Cycle 策略（PyTorch 内置）
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

# PyTorch 的 OneCycleLR
onecycle_scheduler = scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,           # 最大学习率
    total_steps=100,      # 总步数
    pct_start=0.3,        # 预热比例 (30%)
    anneal_strategy='linear',  # 退火策略
    cycle_momentum=True,  # 是否循环动量
    base_momentum=0.85,   # 最小动量
    max_momentum=0.95     # 最大动量
)

print("One Cycle 学习率和动量变化:")
print(f"{'Step':<8} {'LR':<15} {'Momentum':<15}")
print("-" * 40)
for step in range(100):
    onecycle_scheduler.step()
    lr = onecycle_scheduler.get_last_lr()[0]
    # OneCycleLR 自动管理动量
    momentum = optimizer.param_groups[0]['momentum']
    if step % 10 == 0:
        print(f"{step:<8} {lr:<15.8f} {momentum:<15.4f}")

# 对比 One Cycle 和标准训练
print("\n训练对比:")
def train_with_policy(policy_name, use_onecycle=True):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(100, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 10)
    )
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    total_epochs = 50
    if use_onecycle:
        sched = scheduler.OneCycleLR(
            opt, max_lr=0.1, total_steps=total_epochs,
            pct_start=0.3, anneal_strategy='linear'
        )

    X = torch.randn(500, 100)
    y = torch.randint(0, 10, (500,))

    for epoch in range(total_epochs):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if use_onecycle:
            sched.step()

    return loss.item()

standard_loss = train_with_policy("Standard", False)
onecycle_loss = train_with_policy("One Cycle", True)
print(f"  标准训练: {standard_loss:.4f}")
print(f"  One Cycle: {onecycle_loss:.4f}")

# 手动实现简化的 One Cycle
def one_cycle_lr(epoch, total_epochs, max_lr, min_lr=0, warmup_ratio=0.3):
    """手动实现 One Cycle 学习率"""
    warmup_epochs = int(total_epochs * warmup_ratio)
    if epoch < warmup_epochs:
        # 预热阶段：线性增加
        return min_lr + (max_lr - min_lr) * (epoch / warmup_epochs)
    else:
        # 退火阶段：线性减少
        remaining = total_epochs - warmup_epochs
        return max_lr - (max_lr - min_lr) * ((epoch - warmup_epochs) / remaining)

print("\n手动 One Cycle LR:")
total = 50
for e in [0, 10, 15, 25, 35, 45, 49]:
    lr = one_cycle_lr(e, total, 0.1)
    print(f"  epoch {e:2d}: lr={lr:.6f}")

# One Cycle 对训练速度的影响
print("\nOne Cycle 加速效果:")
def train_fast(use_onecycle):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 1))
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.MSELoss()

    X = torch.randn(300, 50)
    y = torch.randn(300, 1)
    total_steps = 30

    if use_onecycle:
        sched = scheduler.OneCycleLR(
            opt, max_lr=0.05, total_steps=total_steps,
            pct_start=0.3
        )

    for step in range(total_steps):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if use_onecycle:
            sched.step()

    return loss.item()

print(f"  标准 30 轮: {train_fast(False):.6f}")
print(f"  One Cycle 30 轮: {train_fast(True):.6f}")
```

## 深度学习关联

- **超收敛发现**：One Cycle 策略最引人注目的发现是"超收敛"——使用合适的学习率范围，模型可以在极少的训练轮数内达到收敛。例如，在 CIFAR-10 上，使用 One Cycle 策略可以在 20-30 个 epoch 内达到原本需要 200+ epoch 才能达到的精度。

- **fast.ai 的核心技术**：One Cycle 策略是 fast.ai 库的核心功能，也是 Leslie Smith 在 fast.ai 工作期间推广的技术。fast.ai 的课程和库广泛使用 One Cycle 策略，使得从业者可以用极少的计算资源快速训练高质量模型。

- **与 Adam 的配合**：虽然 One Cycle 策略最初是为 SGD + Momentum 设计的，但它也可以与 Adam 优化器配合使用。此时只需调度学习率，不需要调度动量。OneCycleLR 在 PyTorch 中支持 `cycle_momentum=False` 选项以适配 Adam。
