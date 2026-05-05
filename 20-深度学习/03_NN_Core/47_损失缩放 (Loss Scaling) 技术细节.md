# 47_损失缩放 (Loss Scaling) 技术细节

## 核心概念

- **损失缩放的定义**：损失缩放（Loss Scaling）是混合精度训练中的关键技术。在反向传播前将损失值乘以一个缩放因子 $S$（如 1024），使梯度在 FP16 可表示范围内，避免下溢为零。反向传播后再将梯度除以 $S$ 恢复原始尺度。
- **梯度下溢问题**：FP16 能表示的最小正数是 $2^{-24} \approx 5.96 \times 10^{-8}$。在深度网络中，大量梯度值远小于这个阈值，在 FP16 下会直接变为 0，导致相应参数无法更新。
- **动态损失缩放**：缩放因子 $S$ 不是固定的，而是动态调整的。如果训练中出现了梯度上溢（inf/NaN），说明 $S$ 太大，将其减半；如果一段时间内没有上溢，可以适当增大 $S$。
- **跳过更新**：当梯度中出现 inf/NaN 时，当前 step 的参数更新被跳过（不执行优化器 step），缩放因子减半，然后重新计算该 step。这确保了训练的稳定性。

## 数学推导

**损失缩放的基本原理**：

标准反向传播：

$$
g = \frac{\partial L}{\partial \theta}
$$

损失缩放后的反向传播：

$$
\tilde{L} = S \cdot L
$$

$$
\tilde{g} = \frac{\partial \tilde{L}}{\partial \theta} = S \cdot \frac{\partial L}{\partial \theta} = S \cdot g
$$

FP16 存储时：$\tilde{g}_{fp16} \approx \tilde{g}$（缩放后的梯度在 FP16 范围内）

更新时：$g_{fp32} = \tilde{g}_{fp16} / S$（恢复原始尺度）

**缩放因子的选择**：

关键约束：$S \cdot |g_{\min}| > \text{FP16}_{\min}$ 且 $S \cdot |g_{\max}| < \text{FP16}_{\max}$

其中 $|g_{\min}|$ 是梯度的最小值，$|g_{\max}|$ 是梯度的最大值。

对于 FP16：$\text{FP16}_{\min} \approx 6 \times 10^{-8}$，$\text{FP16}_{\max} \approx 6.5 \times 10^4$

因此 $S$ 需要满足：$S > 6 \times 10^{-8} / |g_{\min}|$ 且 $S < 6.5 \times 10^4 / |g_{\max}|$

**动态损失缩放算法**：

- 初始化 $S = 2^{16} = 65536$
- 对每个训练步：
   a. 计算缩放后的损失 $\tilde{L} = S \cdot L$
   b. 反向传播计算 $\tilde{g}$
   c. 检查梯度是否包含 inf/NaN
   d. 如果包含 inf/NaN：跳过更新，$S \leftarrow S / 2$
   e. 如果不包含：更新参数，每 $N$ 步 $S \leftarrow S \times 2$

典型参数：$N = 2000$（每 2000 步无 inf 则加倍）

## 直观理解

损失缩放可以类比为"用放大镜观察小物体"——梯度值太小（蚂蚁大小），FP16 的精度无法分辨。损失缩放相当于给蚂蚁拍一张放大的照片（乘以缩放因子），在放大后的尺寸下处理，然后再缩小回真实尺寸。

动态缩放因子类似于"自动调节音量"：如果声音太小听不清（梯度下溢），调高音量（增加 $S$）；如果声音太大导致失真（梯度上溢），调低音量（减小 $S$）。

跳过更新机制就像"安全气囊"——当检测到问题时（梯度中出现 inf/NaN），暂时停止操作（跳过更新），调整缩放因子，然后继续。这防止了一次灾难性的参数更新破坏整个训练过程。

## 代码示例

```python
import torch
import torch.nn as nn

# 演示 FP16 梯度下溢问题
print("FP16 梯度下溢演示:")

# 创建一个需要非常小梯度的场景
x = torch.tensor([1000.0], requires_grad=True)
loss = x * 1e-10  # 非常小的损失
loss.backward()
print(f"  梯度（FP32）: {x.grad.item():.15e}")

try:
    grad_fp16 = x.grad.half()
    print(f"  梯度（FP16）: {grad_fp16.item():.15e} (下溢为 0!)")
except:
    print(f"  梯度（FP16）: 下溢为 0")

# 使用损失缩放解决下溢
S = 1024.0
x2 = torch.tensor([1000.0], requires_grad=True)
loss2 = S * x2 * 1e-10  # 缩放后的损失
loss2.backward()
print(f"\n损失缩放后梯度（FP32）: {x2.grad.item():.15e}")
print(f"损失缩放后梯度（FP16）: {x2.grad.half().item():.15e} (不再下溢)")
print(f"恢复尺度后梯度: {x2.grad.half().item() / S:.15e}")

# 损失缩放对训练的影响
print("\n损失缩放对训练的影响:")
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(100, 256), nn.ReLU(),
    nn.Linear(256, 256), nn.ReLU(),
    nn.Linear(256, 1)
)

X = torch.randn(100, 100)
y = torch.randn(100, 1)

# FP32 训练（基准）
opt_fp32 = torch.optim.Adam(model.parameters(), lr=0.001)
losses_fp32 = []
for epoch in range(50):
    pred = model(X)
    loss = ((pred - y) ** 2).mean()
    opt_fp32.zero_grad()
    loss.backward()
    opt_fp32.step()
    losses_fp32.append(loss.item())

# 模拟损失缩放的梯度稳定性
print("演示 GradScaler API:")
scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

if scaler:
    print(f"  初始缩放因子: {scaler.get_scale()}")
else:
    print("  CPU 模式，演示 GradScaler 用法:")
    print("  scaler = torch.cuda.amp.GradScaler()")
    print("  scaler.scale(loss).backward()")
    print("  scaler.step(optimizer)")
    print("  scaler.update()")

# 手动实现动态损失缩放
print("\n手动动态损失缩放:")
class DynamicLossScaler:
    def __init__(self, init_scale=2**16, scale_factor=2.0, scale_window=2000):
        self.scale = init_scale
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.steps_since_update = 0

    def scale_loss(self, loss):
        return loss * self.scale

    def has_inf_or_nan(self, tensor):
        return torch.isinf(tensor).any() or torch.isnan(tensor).any()

    def update(self, grads):
        has_inf = any(self.has_inf_or_nan(g) for g in grads if g is not None)
        if has_inf:
            self.scale /= self.scale_factor
            self.steps_since_update = 0
            return False  # 跳过更新
        else:
            self.steps_since_update += 1
            if self.steps_since_update >= self.scale_window:
                self.scale *= self.scale_factor
                self.steps_since_update = 0
            return True  # 正常更新

scaler_manual = DynamicLossScaler()
print(f"  初始缩放因子: {scaler_manual.scale}")
print(f"  缩放因子加倍频率: 每 {scaler_manual.scale_window} 步")
print(f"  遇到 inf 时: 缩放因子减半，跳过更新")

# BF16（不需要损失缩放）
print("\nBF16 vs FP16:")
print("  FP16: 5 位指数, 范围约 [6e-8, 6.5e4]")
print("  BF16: 8 位指数, 范围约 [1e-38, 3e38] (和 FP32 相同)")
print("  BF16 不需要损失缩放，因为范围足够大")
```

## 深度学习关联

- **混合精度训练的标配**：损失缩放是所有混合精度训练框架的标准组件。PyTorch 的 `torch.cuda.amp.GradScaler`、TensorFlow 的 `tf.train.experimental.DynamicLossScale`、NVIDIA 的 APEX AMP 都实现了损失缩放。默认初始缩放因子通常为 $2^{16}=65536$。
- **BF16 消除了损失缩放需求**：Google 的 TPU 和 NVIDIA A100+ 硬件支持的 BF16 格式具有与 FP32 相同的指数范围，因此不需要损失缩放。这简化了混合精度训练的实现。对于新硬件，BF16 正在取代 FP16 成为混合精度训练的首选格式。
- **跳过更新的影响**：频繁的跳过更新（由梯度上溢导致）会减慢训练速度。正常情况下，跳过更新应极少发生（少于 1% 的步数）。如果频繁跳过，可能表示学习率过大或缩放因子初始值不合适，需要调整。
