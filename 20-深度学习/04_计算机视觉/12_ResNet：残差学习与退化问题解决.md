# 13_ResNet：残差学习与退化问题解决

## 核心概念

- **退化问题（Degradation Problem）**：当网络深度增加时，训练准确率出现饱和甚至下降，并非由过拟合引起（训练误差同时升高）。这表明深层网络难以优化。
- **残差学习（Residual Learning）**：不直接学习目标映射 $\mathcal{H}(x)$，而是学习残差映射 $\mathcal{F}(x) = \mathcal{H}(x) - x$，即 $y = \mathcal{F}(x) + x$。当最优映射接近恒等映射时，让网络学习"零残差"比学习恒等映射更容易。
- **快捷连接（Shortcut Connection）**：跳过一层或多层直接将输入加到输出上，不增加额外参数和计算量。当维度不匹配时使用 $1\times1$ 卷积调整。
- **Bottleneck结构**：对于深层ResNet（50层以上），使用 $1\times1 \rightarrow 3\times3 \rightarrow 1\times1$ 的三层瓶颈结构代替两层 $3\times3$，先降维再升维，减少计算量。
- **批归一化（Batch Normalization）**：每个卷积层后紧跟BN，加速训练、缓解梯度消失、提供正则化效果。
- **网络深度突破**：ResNet成功训练了152层的网络（VGG的10倍深度），在ImageNet上top-5错误率降至3.57%，超过人类水平。

## 数学推导

**残差块的前向传播：**
$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

其中 $\mathcal{F}(x, \{W_i\})$ 是残差映射。对于两层残差块：
$$
\mathcal{F} = W_2 \sigma(W_1 x + b_1) + b_2
$$
其中 $\sigma$ 是ReLU激活函数。

**残差块的反向传播优势：**
设损失函数为 $L$，输出为 $y = \mathcal{F}(x) + x$，则：
$$
\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right)
$$

梯度中出现了常数项"1"，保证了梯度可以无损地直接回传到浅层，有效缓解了梯度消失。即使 $\frac{\partial \mathcal{F}}{\partial x} \to 0$，梯度仍然可以通过"1"的路径传播。

**维度匹配的三种快捷连接方式：**
- (A) 恒等映射 + 维度补齐（零填充），无额外参数
- (B) 维度不匹配时使用投影 $1\times1$ 卷积
- (C) 所有快捷连接都使用 $1\times1$ 卷积（全投影）

## 直观理解

可以把残差学习想象成"增量学习"：传统网络就像一个学生要从零开始学习一个完整的函数（$y = f(x)$），而残差网络则是让学生先"复制答案"（恒等映射 $y = x$），再学习"还需要修正什么"（残差 $\mathcal{F}(x)$）。显然，在大多数情况下，"复制答案再微调"比"从零开始"容易得多。

梯度传播中的"1"就像一条高速公路——整个网络中始终有一条"快速通道"让梯度直接流回浅层，避免了经过多层非线性变换后梯度衰减的问题。这让训练超深网络成为可能。

## 代码示例

```python
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """ResNet基本残差块 (用于ResNet-18/34)"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # 残差连接（核心）
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    """瓶颈残差块 (用于ResNet-50/101/152)"""
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

# 对比：有/无残差连接的梯度
model_with_residual = BasicBlock(64, 64)
x = torch.randn(1, 64, 56, 56, requires_grad=True)
out = model_with_residual(x)
loss = out.sum()
loss.backward()
print(f"有残差连接的梯度范数: {x.grad.norm().item():.4f}")
```

## 深度学习关联

- **突破深度瓶颈**：ResNet成功训练了152层网络，证明了深度网络的可训练性。此后的DenseNet（密集连接）、ResNeXt（分组残差）、EfficientNet等都建立在残差学习的基础之上。
- **梯度流动的通用设计模式**：残差连接的"梯度高速公路"思想被广泛应用于Transformer（Pre-LN残差连接）、扩散模型（残差块堆叠）等非CNN架构中，成为深度学习最基础的组件之一。
- **预训练-微调范式**：ResNet凭借其强大的泛化能力和层次化特征表示，成为最常用的视觉骨干网络，广泛用于目标检测（Faster R-CNN的backbone）、语义分割（DeepLab的backbone）、视频理解等下游任务。
