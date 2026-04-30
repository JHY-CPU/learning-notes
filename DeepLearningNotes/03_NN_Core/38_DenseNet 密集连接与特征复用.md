# 38 DenseNet 密集连接与特征复用

## 核心概念

- **DenseNet 的定义**：DenseNet（Densely Connected Convolutional Network）将每一层的输出连接到后续所有层的输入。第 $l$ 层接收之前所有 $l-1$ 层的输出作为输入：$x_l = H_l([x_0, x_1, \ldots, x_{l-1}])$，其中 $[\cdot]$ 表示特征图的拼接（concatenation）。

- **特征复用的思想**：与 ResNet 的"求和"不同，DenseNet 使用"拼接"来组合特征。这鼓励特征复用——每一层不需要重新学习前面层已经提取的特征，只需学习少量新的特征。这使网络参数更少、效率更高。

- **缓解梯度消失**：由于每一层都直接连接到后续所有层，梯度可以沿着这些密集连接直接流回早期层。这加强了梯度流动，缓解了梯度消失问题，类似于残差连接但更强。

- **隐式深度监督**：DenseNet 的密集连接结构天然地提供了"深度监督（Deep Supervision）"的效果——分类器的梯度可以直接影响所有层，包括早期层。这使得训练非常深的网络成为可能。

## 数学推导

**DenseNet 的层间连接**：

对于第 $l$ 层（假设有 $L$ 层），输入是所有前面层特征图的拼接：

$$
x_l = H_l([x_0, x_1, x_2, \ldots, x_{l-1}])
$$

其中 $H_l$ 是复合函数，通常包含 BN → ReLU → Conv（3x3）。

**特征图数量的增长（Growth Rate）**：

假设每层输出 $k$ 个新特征图（$k$ 称为增长率，通常取 12、24、32）。第 $l$ 层的输入通道数为 $k_0 + k \times (l-1)$，其中 $k_0$ 是初始输入通道数。

这种设计使得 DenseNet 非常"窄"（每层只输出少量特征图），通过拼接前面积累大量特征图。与 ResNet 形成对比——ResNet 通常使用数百个通道，而 DenseNet 使用数十个。

**梯度流动分析**：

损失对第 $l$ 层输入的梯度：

$$
\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \frac{\partial x_L}{\partial x_l} + \sum_{i=l}^{L-1} \frac{\partial L}{\partial x_{i+1}} \cdot \frac{\partial H_i(x_i)}{\partial x_l}
$$

由于密集连接，梯度可以通过多条路径直接从 $x_L$ 流到 $x_l$，每条路径提供直接的梯度信息。这比 ResNet 的单条恒等路径更强。

**参数效率**：

DenseNet 的参数量分析（不考虑 bottleneck）：

$$
\text{Params} = \sum_{l=1}^{L} (k_0 + k(l-1)) \times k \times 3 \times 3
$$

作为对比，等效的 ResNet 参数量更大。例如，DenseNet-121 只有约 8M 参数，但可以达到 ResNet-50（25M 参数）相近的性能。

## 直观理解

DenseNet 的密集连接可以类比为"无领导小组讨论"：每层（参与者）都能看到之前所有层（其他参与者）的输出（发言内容）。这种"全透明"的信息共享机制使得知识在层间高效流动，每一层都可以基于所有前层的输出做决策。

特征复用就像是"搭积木"：每一层只需要添加几块新的积木（少量新特征），然后把新的积木块添加到已有的积木堆上。之前的积木被重复利用，不需要重新搭建。这大大节省了"积木"（参数）。

与 ResNet 的对比：ResNet 是"渐进式改进"——每一层对输入做一次修改，但修改幅度较小；DenseNet 是"聚合式构建"——每一层都在现有特征的基础上添加新的特征，逐渐丰富特征表示。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# DenseNet 核心组件

class Bottleneck(nn.Module):
    """DenseNet 的 Bottleneck 层: BN-ReLU-Conv(1x1)-BN-ReLU-Conv(3x3)"""
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        # 1x1 卷积压缩通道数（bottleneck）
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, 1, bias=False)
        # 3x3 卷积输出 growth_rate 个新特征图
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, 3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        return out

class Transition(nn.Module):
    """过渡层: 1x1 卷积降维 + 2x2 平均池化"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = F.relu(self.bn(x))
        x = self.conv(x)
        return F.avg_pool2d(x, 2)

class DenseBlock(nn.Module):
    """密集连接块"""
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(Bottleneck(in_channels + i * growth_rate, growth_rate))

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)

# 简化的 DenseNet
class SimpleDenseNet(nn.Module):
    def __init__(self, growth_rate=12, block_layers=[6, 12, 24, 16], num_classes=10):
        super().__init__()
        num_channels = 2 * growth_rate

        # 初始卷积
        self.conv1 = nn.Conv2d(3, num_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)

        # Dense Blocks
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, n_layers in enumerate(block_layers):
            block = DenseBlock(n_layers, num_channels, growth_rate)
            self.blocks.append(block)
            num_channels += n_layers * growth_rate

            if i < len(block_layers) - 1:
                trans = Transition(num_channels, num_channels // 2)
                self.transitions.append(trans)
                num_channels = num_channels // 2

        # 分类头
        self.bn_final = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block, trans in zip(self.blocks, self.transitions):
            x = block(x)
            x = trans(x)
        x = self.blocks[-1](x)  # 最后一个 block 后没有 transition
        x = F.relu(self.bn_final(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 演示 DenseNet 的特征复用
torch.manual_seed(42)

model = SimpleDenseNet(growth_rate=12, block_layers=[4, 4, 4], num_classes=10)
x = torch.randn(2, 3, 32, 32)
out = model(x)
print(f"DenseNet 输出 shape: {out.shape}")
total_params = sum(p.numel() for p in model.parameters())
print(f"DenseNet 参数量: {total_params:,}")

# 对比 DenseNet 和等参数量的普通网络
# DenseNet 参数主要集中在最后的分类层和 transition
print("\nDenseNet 各组件参数量:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"  {name}: {param.numel():,} 参数")

# 演示特征图数量的增长
print("\n特征图数量变化:")
dummy = torch.randn(1, 3, 32, 32)
h = F.relu(model.bn1(model.conv1(dummy)))
print(f"  初始: {h.shape[1]} 通道")
for i, block in enumerate(model.blocks):
    h = block(h)
    print(f"  Block {i}: {h.shape[1]} 通道")
    if i < len(model.transitions):
        h = model.transitions[i](h)
        print(f"  Transition {i}: {h.shape[1]} 通道")
```

## 深度学习关联

- **参数效率的突破**：DenseNet 以其高参数效率著称。DenseNet-121（8M 参数）可以达到与 ResNet-50（25M 参数）相当的 ImageNet Top-1 准确率。这使得 DenseNet 在计算资源受限的场景中特别有吸引力。

- **密集连接的计算代价**：DenseNet 的缺点在于推理时的计算和内存开销较大。虽然参数少，但由于特征图拼接导致中间特征图的通道数很大，显存占用高。后来的优化（如 Memory-efficient DenseNet）解决了这一问题。

- **对后续架构的影响**：DenseNet 的密集连接思想影响了后续架构的设计。CSPNet（Cross Stage Partial Network）将 DenseNet 的密集连接进行了裁剪以降低计算量。CondenseNet 在 DenseNet 基础上引入了可学习的分组卷积，进一步提升了效率。
