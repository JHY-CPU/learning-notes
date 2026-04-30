# 43 Cosine Annealing 余弦退火策略

## 核心概念

- **Cosine Annealing 定义**：Cosine Annealing（余弦退火）按照余弦函数的形式衰减学习率。学习率从初始值 $\eta_0$ 开始，沿余弦曲线平滑下降到最小值 $\eta_{\min}$。公式为 $\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})(1 + \cos(\frac{t}{T}\pi))$。

- **平滑衰减**：与 StepLR 的阶梯式衰减不同，Cosine Annealing 的衰减是连续且平滑的。这种平滑衰减避免了学习率突变带来的训练扰动，使模型能够更稳定地收敛。

- **热重启（SGDR）**：Cosine Annealing 的扩展版本 SGDR（Stochastic Gradient Descent with Restarts）周期性地重启学习率到初始值。每次重启相当于"跳出当前局部区域并开始新的搜索"，有助于逃离局部极小值。

- **无超参数调优**：Cosine Annealing 不需要手动设定衰减时刻（不像 MultiStepLR 需要 milestones）。唯一需要确定的是总训练轮数 $T$ 和初始学习率 $\eta_0$，使用非常简单。

## 数学推导

**Cosine Annealing 基本形式**：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_0 - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

其中：
- $\eta_0$：初始学习率
- $\eta_{\min}$：最小学习率（通常设为 0）
- $t$：当前步数（或 epoch 数）
- $T$：总步数

当 $\eta_{\min} = 0$ 时简化为：

$$
\eta_t = \frac{\eta_0}{2}\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)
$$

**SGDR（带热重启的余弦退火）**：

SGDR 在每次重启时将 $t$ 重置为 0，并将 $\eta_0$ 恢复为初始值。第 $i$ 次重启周期的学习率：

$$
\eta_t^{(i)} = \eta_{\min} + \frac{1}{2}(\eta_0^{(i)} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_i}\pi\right)\right)
$$

其中 $T_i$ 可以是固定值（等周期）或递增的值（如 $T_i = T_0 \cdot 2^i$）。

**衰减曲线对比**：

在 $[0, T]$ 区间内：
- StepLR：阶梯函数，突变
- ExponentialLR：指数函数，初始下降快后期下降慢
- Cosine Annealing：先慢后快再慢的 S 形衰减

Cosine Annealing 的独特之处在于三个阶段：
1. **初始阶段**（$t \approx 0$）：学习率下降缓慢，保持较大值探索
2. **中间阶段**（$t \approx T/2$）：学习率快速下降
3. **最终阶段**（$t \approx T$）：学习率下降减缓，精细微调

## 直观理解

Cosine Annealing 的学习率变化就像"日落"——太阳下山时，先是缓慢变暗（初始阶段），然后快速变暗（中间阶段），最后又缓慢地进入完全的黑暗（最终阶段）。这种"先慢后快再慢"的模式在视觉上正好是一条余弦曲线。

为什么这种模式有效？训练初期需要足够的时间在高学习率下探索（所以学习率下降慢），然后快速进入精细调优阶段，最后在最低学习率下做最后收敛。

热重启（SGDR）的直觉是"退一步进两步"——每隔一段时间将学习率跳回高值，让模型跳出当前的"局部舒适区"，探索新的参数空间。这类似于模拟退火中的"升温"过程，有助于逃离局部极小值。重启后的模型通常能在更短的时间内达到更好的性能。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as scheduler

# Cosine Annealing 基本使用
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 基本 Cosine Annealing
cosine_scheduler = scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0)

# SGDR (带热重启)
sgdr_scheduler = scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=30, T_mult=2, eta_min=0.001
)

print("Cosine Annealing 学习率变化:")
print(f"{'Epoch':<10} {'Cosine':<15} {'SGDR (T0=30, T_mult=2)':<15}")
print("-" * 50)
for epoch in range(100):
    cosine_scheduler.step()
    sgdr_scheduler.step(epoch)  # SGDR 需要传入当前 epoch
    lr_cos = cosine_scheduler.get_last_lr()[0]
    lr_sgdr = sgdr_scheduler.get_last_lr()[0]
    if epoch % 10 == 0 or epoch == 30 or epoch == 31:
        print(f"{epoch:<10} {lr_cos:<15.6f} {lr_sgdr:<15.6f}")

# Cosine Annealing vs StepLR 训练对比
print("\n训练对比:")
def train_with_scheduler(scheduler_constructor, scheduler_kwargs, name, epochs=100):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(50, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 10)
    )
    opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = scheduler_constructor(opt, **scheduler_kwargs)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(500, 50)
    y = torch.randint(0, 10, (500,))

    for epoch in range(epochs):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

    return loss.item()

cos_loss = train_with_scheduler(
    scheduler.CosineAnnealingLR, {"T_max": 100, "eta_min": 0}, "Cosine")
step_loss = train_with_scheduler(
    scheduler.StepLR, {"step_size": 30, "gamma": 0.1}, "StepLR")

print(f"  Cosine Annealing: {cos_loss:.4f}")
print(f"  StepLR: {step_loss:.4f}")

# 手动实现 Cosine Annealing
def cosine_annealing_lr(epoch, total_epochs, lr_init, lr_min=0):
    """手动计算余弦退火学习率"""
    return lr_min + 0.5 * (lr_init - lr_min) * (1 + torch.cos(
        torch.tensor(epoch / total_epochs * torch.pi)
    ))

print("\n手动计算的学习率:")
for epoch in [0, 25, 50, 75, 99]:
    lr = cosine_annealing_lr(epoch, 100, 0.1)
    print(f"  epoch {epoch}: lr={lr:.6f}")

# SGDR 的热重启效果
print("\nSGDR 热重启周期:")
T_0 = 20
T_mult = 2
current_T = T_0
current_epoch = 0
for restart in range(3):
    print(f"  重启 {restart+1}: 周期长度 = {current_T}, 起始 epoch = {current_epoch}")
    current_epoch += current_T
    current_T = current_T * T_mult
```

## 深度学习关联

- **现代深度学习的首选调度器**：Cosine Annealing 是许多现代深度学习训练的标准选择。它在不需要额外超参数调优的情况下，通常能达到或超过手动设计的 StepLR/MultiStepLR 的效果。许多 Kaggle 竞赛方案使用 Cosine Annealing。

- **与 Warmup 的组合**：Warmup + Cosine Annealing 的组合已成为 NLP 模型（BERT、GPT）训练的标准配置。训练开始时先用 Warmup 线性增加学习率到最大值，然后用 Cosine Annealing 衰减到零。这种组合在 Transformer 训练中效果最佳。

- **SGDR 在模型集成中的应用**：SGDR 的周期性重启可以收集到多个性能良好的模型检查点。这些检查点对应于不同训练周期的局部极小值，可以用于模型集成（Snapshot Ensembling），提升最终性能而不增加额外训练成本。
