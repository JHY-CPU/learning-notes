# 14_ResNeXt：基数 (Cardinality) 的概念

## 核心概念

- **基数（Cardinality）**：ResNeXt提出的核心概念，指同一层中独立分支（变换路径）的数量。基数是除了深度和宽度之外的第三个可扩展维度。
- **分组卷积实现**：ResNeXt使用分组卷积（groups）高效实现多分支并行结构。每个分支共享相同的拓扑结构（$1\times1 \rightarrow 3\times3 \rightarrow 1\times1$），通过分组数控制基数。
- **等价架构**：ResNeXt的构建块有三种等价实现形式：(a) 多分支并行 (b) 分组卷积 (c) 通道分割+分组卷积。三者数学等价，但分组卷积的实现效率最高。
- **维度扩展"三驾马车"**：ResNeXt系统比较了三种增加模型容量的方式——加深（深度）、加宽（宽度）和增加基数（cardinality），发现增加基数在相同计算量下收益最高。
- **VGG/ResNet的规整设计原则**：ResNeXt继承并扩展了"规整设计"理念——所有分支具有相同的拓扑结构，简化了超参数搜索，使网络设计更系统化。
- **与Inception的对比**：Inception的不同分支使用不同的卷积核尺寸，而ResNeXt的所有分支使用完全相同的结构，更简单、更可扩展。

## 数学推导

**ResNeXt构建块的计算公式：**

将输入 $x$ 通过 $C$（基数）个相同拓扑的变换 $\mathcal{T}_i$，然后拼接求和：
$$
y = x + \sum_{i=1}^{C} \mathcal{T}_i(x)
$$

其中每个 $\mathcal{T}_i$ 包含三个子层：$1\times1$（降维）$\rightarrow$ $3\times3$（空间卷积，分组）$\rightarrow$ $1\times1$（升维）。

**三种等价实现形式的参数量对比：**

假设输入/输出通道均为 $D$，中间通道为 $d$，基数 $C$：

形式 (a) 多分支：$C \times (1\times1 D \to d + 3\times3 d \to d + 1\times1 d \to D)$
形式 (b) 分组卷积：$1\times1 D \to (C \cdot d) + 3\times3\text{ groups=C } (C \cdot d) \to (C \cdot d) + 1\times1 (C \cdot d) \to D$

两者参数量相同：$(D \times d \times 1 \times 1) \times C + (d \times d \times 9 / C) \times C + (d \times D \times 1 \times 1) \times C$

**实验结论（ImageNet）：**
在相同计算量（约4G FLOPs）下：
- ResNet-50（基数=1, 宽度=4d）: 76.0% top-1
- ResNeXt-50（基数=32, 宽度=4d）: 77.8% top-1 (+1.8%)
- ResNeXt-50（基数=64, 宽度=4d）: 78.1% top-1 (+2.1%)

增加基数比加倍深度或加倍宽度带来的增益更大。

## 直观理解

基数可以理解为"观点的多样性"。传统ResNet中只有一个"专家"（一个残差分支）来做特征变换，而ResNeXt引入了多个"独立专家"（32个或64个分支），每个专家从不同的角度分析输入特征，最后汇总所有人的意见。这就像在决策时咨询多个领域的专家而非依赖一个"万能专家"——多样的观点往往带来更准确的判断。

分组卷积则像是"并行办公"——32个专家同时工作（32组），每组内只有几十人的小团队，比一个拥有上万人（宽层）的大部门效率更高。

## 代码示例

```python
import torch
import torch.nn as nn

class ResNeXtBlock(nn.Module):
    """ResNeXt 构建块 (基数=32)"""
    def __init__(self, in_channels, out_channels, cardinality=32, stride=1):
        super().__init__()
        # 中间通道数 = 输出通道数 / 2 (标准设置)
        mid_channels = out_channels // 2

        self.conv1 = nn.Conv2d(in_channels, mid_channels * cardinality, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels * cardinality)
        # 分组卷积：groups=cardinality 是关键
        self.conv2 = nn.Conv2d(
            mid_channels * cardinality,
            mid_channels * cardinality,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,  # 基数通过分组数实现
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels * cardinality)
        self.conv3 = nn.Conv2d(mid_channels * cardinality, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 维度匹配
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

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

# 验证不同基数的参数量差异
for C in [1, 4, 8, 32]:
    block = ResNeXtBlock(128, 256, cardinality=C)
    params = sum(p.numel() for p in block.parameters())
    print(f"基数={C:2d}, 参数量={params:,}")
```

## 深度学习关联

- **深度、宽度、基数三轴缩放**：ResNeXt提出的"基数"作为第三维度拓展了网络设计空间。后续的EfficientNet系统研究了深度、宽度和分辨率的复合缩放，将这一思想推广到更一般的维度。
- **分组卷积的复兴**：在AlexNet之后，分组卷积因硬件限制放宽而被遗忘，ResNeXt重新发掘了其价值，证明了适度的分组可以提升模型质量。这直接影响了ShuffleNet、MobileNet V2等轻量级网络的分组设计。
- **Transformer中的多头注意力**：多头注意力机制中的"头"本质上也是一种基数——每个头独立学习不同的注意力模式，最后拼接融合。ResNeXt的分组卷积设计思想与多头注意力有着异曲同工之妙，反映了"分组处理+融合"这一通用模式在深度学习中的广泛适用性。
