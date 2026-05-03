# 36_Group Normalization：小 Batch 下的替代方案

## 核心概念

- **Group Normalization 的定义**：Group Normalization（GN）将通道分成若干组，在每个组内独立计算均值和方差进行标准化。它是 BN 和 LN 的折中方案——在通道维度上分组，每组包含若干通道。
- **小 batch 时的优势**：当 batch size 很小时（如 1, 2, 4），BN 的统计量估计极不准确，导致训练不稳定甚至失败。GN 不依赖 batch 维度，无论 batch size 多大都能稳定工作。
- **BN、LN、IN、GN 的统一视角**：四种归一化方法可以统一为"在某个维度子集上计算均值和方差"的操作。BN 在 batch 维度，LN 在通道维度，IN 在通道+空间维度，GN 在分组通道维度。
- **分组数的选择**：默认分组数 $G = 32$（如果通道数不足 32 则 $G = \text{num\_channels}$）。$G = 1$ 时 GN 退化为 LN，$G = C$ 时退化为 IN。GN 是 LN 和 IN 的中间状态。

## 数学推导

**GN 前向传播**：

对输入 $x \in \mathbb{R}^{N \times C \times H \times W}$，将 $C$ 个通道分为 $G$ 组，每组有 $C/G$ 个通道。

对每组 $g$：

$$
\mathcal{S}_g = \{(c, h, w) | c \in [gC/G, (g+1)C/G), h \in [1,H], w \in [1,W]\}
$$

$$
\mu_{ng} = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in \text{group }g} \sum_{h,w} x_{nchw}
$$

$$
\sigma_{ng}^2 = \frac{1}{(C/G) \cdot H \cdot W} \sum_{c \in \text{group }g} \sum_{h,w} (x_{nchw} - \mu_{ng})^2
$$

$$
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{ng}}{\sqrt{\sigma_{ng}^2 + \epsilon}}
$$

$$
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
$$

**四种归一化的统一视角**：

$$
\mu_{n} = \frac{1}{|\mathcal{S}|} \sum_{(c,h,w) \in \mathcal{S}} x_{nchw}
$$

其中 $\mathcal{S}$ 的定义：

| 方法 | $\mathcal{S}$ 包含的维度 | 参数量 |
|------|------------------------|--------|
| BN | $\{n \in N\}$ | $O(C)$ |
| LN | $\{c \in C\}$ | $O(C)$ |
| IN | $\{c \in C, h \in H, w \in W\}$ | $O(C)$ |
| GN | $\{c \in C/G, h \in H, w \in W\}$ | $O(C)$ |

**GN 作为 BN、LN、IN 的插值**：

- $G = 1$：GN = LN（所有通道为一组）
- $G = C$：GN = IN（每个通道独立一组）
- $G$ 取中间值：在 LN 和 IN 之间平滑插值

## 直观理解

Group Normalization 可以理解为"用小团体替代整体"：与其统计整个班级（LN）或整个学校（BN）的情况，不如把特征通道分成几个小组，在组内标准化。这就像把学生分成小组，每组独立计算平均水平和差异程度。

当 batch size 很小时（如 batch=1），BN 的统计量退化为单个样本的统计量，失去了"跨样本标准化"的意义，实际上退化为了 IN。但 IN 对每个通道独立标准化，过于细粒度，有时会损失特征间的相关性信息。GN 在组内保留了多个通道的联合统计，平衡了细粒度和信息保留。

$G$ 是一个关键超参数。$G$ 太小（接近 LN）会弱化每组内通道之间的差异化；$G$ 太大（接近 IN）会丢失通道间的联合统计信息。论文建议默认 $G=32$，这是一个在实践中表现良好的平衡点。

## 代码示例

```python
import torch
import torch.nn as nn

# 手动实现 Group Normalization
class GroupNormManual(nn.Module):
    def __init__(self, num_channels, num_groups=32, eps=1e-5, affine=True):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.affine = affine

        if num_channels % num_groups != 0:
            raise ValueError(f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})")

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # x: (N, C, H, W)
        N, C, H, W = x.shape
        G = self.num_groups
        group_size = C // G

        # reshape: (N, G, group_size, H, W)
        x = x.view(N, G, group_size, H, W)
        mean = x.mean(dim=(2, 3, 4), keepdim=True)  # (N, G, 1, 1, 1)
        var = x.var(dim=(2, 3, 4), keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()

        # reshape back: (N, C, H, W)
        x_norm = x_norm.view(N, C, H, W)

        if self.affine:
            x_norm = x_norm * self.gamma.view(1, C, 1, 1) + self.beta.view(1, C, 1, 1)
        return x_norm

# 验证与 PyTorch 内置 GN 的一致性
torch.manual_seed(42)
x = torch.randn(4, 64, 16, 16)

gn_manual = GroupNormManual(64, 8)
gn_pytorch = nn.GroupNorm(8, 64)
gn_pytorch.weight.data = gn_manual.gamma.data.clone()
gn_pytorch.bias.data = gn_manual.beta.data.clone()

out_manual = gn_manual(x)
out_pytorch = gn_pytorch(x)
print(f"GN 输出一致: {torch.allclose(out_manual, out_pytorch, atol=1e-5)}")

# 演示 BN 在小 batch 下的失效
print("\nBatch size 对归一化方法的影响 (batch=2):")
batch = 2
x = torch.randn(batch, 64, 32, 32)

bn = nn.BatchNorm2d(64)
gn = nn.GroupNorm(8, 64)

bn_out = bn(x)
gn_out = gn(x)

print(f"  BN 输出 std: {bn_out.std():.4f}")
print(f"  GN 输出 std: {gn_out.std():.4f}")

# 演示不同分组数的影响
print("\n不同分组数对输出的影响:")
x = torch.randn(4, 32, 8, 8)
for G in [1, 2, 4, 8, 16, 32]:
    gn = nn.GroupNorm(G, 32)
    out = gn(x)
    print(f"  G={G:2d}: 输出 mean={out.mean():.4f}, std={out.std():.4f}, "
          f"每组通道数={32//G}")

# 训练对比: 小 batch 下 BN vs GN
print("\n小 batch 训练对比 (batch=2):")
def train_with_norm(norm_layer, name):
    torch.manual_seed(42)
    if norm_layer == 'bn':
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
        )
    else:
        model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.GroupNorm(8, 32), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(16, 64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 10)
        )

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(20, 3, 32, 32)
    y = torch.randint(0, 10, (20,))

    for epoch in range(100):
        # 使用 batch=2
        for i in range(0, 20, 2):
            batch_x = X[i:i+2]
            batch_y = y[i:i+2]
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        out = model(X)
        acc = (out.argmax(1) == y).float().mean()
    return acc.item()

print(f"  BN: acc={train_with_norm('bn', 'BN'):.4f}")
print(f"  GN: acc={train_with_norm('gn', 'GN'):.4f}")
```

## 深度学习关联

- **检测和分割任务的首选**：Group Normalization 在目标检测和语义分割等任务中表现优异，因为这些任务通常需要小 batch 训练（受 GPU 内存限制，batch size 可能只有 1-4）。Mask R-CNN 使用 GN 替代 BN 后性能稳定提升，特别是 batch size 很小时。
- **视频理解中的优势**：在视频理解任务中，batch size 通常很小（因为每个视频帧占用大量显存），GN 是比 BN 更好的选择。GN 在 Video ResNet、I3D 等视频模型中广泛使用。
- **GN 与 BN 的互补使用**：一种常见策略是在大 batch 预训练时使用 BN，在下游任务微调时切换到 GN（因为下游任务的 batch size 通常较小）。这种两阶段策略同时利用了 BN 的计算效率和 GN 的小 batch 稳定性。
