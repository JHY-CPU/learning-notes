# 10_AlexNet：ReLU、Dropout 与 GPU 训练

## 核心概念

- **AlexNet 的历史意义**：2012 年 ImageNet 竞赛冠军（Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton），将图像分类 Top-5 错误率从 26% 降至 15.3%，开启了深度学习时代。
- **ReLU 激活函数**：AlexNet 首次在大型 CNN 中使用 ReLU（Rectified Linear Unit）$f(x) = \max(0, x)$，相比 Tanh/Sigmoid 大幅缓解梯度消失问题，训练速度提升数倍。
- **Dropout 正则化**：在训练过程中随机丢弃 50% 的神经元（将其输出置零），迫使网络学习冗余表示，有效防止过拟合。
- **GPU 并行训练**：将网络拆分为两部分在两个 GPU（NVIDIA GTX 580）上并行训练，GPU 间只在特定层通信。这既是硬件限制下的妥协，也开创了分布式深度学习训练的先河。
- **局部响应归一化（LRN）**：在相邻通道间做局部归一化，模拟神经生物学的"侧抑制"现象，增强模型泛化能力（后续被 BatchNorm 取代）。
- **重叠池化（Overlapping Pooling）**：使用 $3\times3$ 池化窗口、步长=2，相邻窗口有重叠区域，相比 $2\times2$ 无重叠池化提升约 0.3% 精度。
- **数据增强**：使用随机裁剪（$224\times224$ 从 $256\times256$ 中裁剪）和水平翻转，以及 PCA 颜色增强（改变 RGB 通道强度），使训练数据扩大 2048 倍。

## 数学推导

**AlexNet 架构参数：**

| 层 | 操作 | 核/尺寸 | 输出尺寸 | 参数量 |
|---|---|---|---|---|
| Conv1 | Conv + ReLU + LRN + MaxPool | $11\times11/4$, 96 | $55\times55\times96$ | 35K |
| Conv2 | Conv + ReLU + LRN + MaxPool | $5\times5/1$, 256 (pad=2) | $27\times27\times256$ | 307K |
| Conv3 | Conv + ReLU | $3\times3/1$, 384 (pad=1) | $13\times13\times384$ | 885K |
| Conv4 | Conv + ReLU | $3\times3/1$, 384 (pad=1) | $13\times13\times384$ | 1.3M |
| Conv5 | Conv + ReLU + MaxPool | $3\times3/1$, 256 (pad=1) | $6\times6\times256$ | 885K |
| FC6 | FC + ReLU + Dropout | 4096 | 4096 | 38M |
| FC7 | FC + ReLU + Dropout | 4096 | 4096 | 16.8M |
| FC8 | FC + Softmax | 1000 | 1000 | 4.1M |

**ReLU 激活函数及其梯度：**
$$
\text{ReLU}(x) = \max(0, x)
$$
$$
\frac{d}{dx}\text{ReLU}(x) = \begin{cases}
1 & x > 0 \\
0 & x \leq 0
\end{cases}
$$

对比 Tanh 的梯度 $\frac{d}{dx}\tanh(x) = 1 - \tanh^2(x)$ 在饱和区趋近于0，ReLU 在正半轴的恒等梯度避免了梯度消失问题。

**Dropout 训练与推理：**
训练时：$y = r \odot h$，其中 $r_i \sim \text{Bernoulli}(p=0.5)$
推理时：$y = p \cdot h$（乘以保留概率以保持期望一致）

## 直观理解

AlexNet 的成功可以归结为三个关键创新。ReLU 像是一个"信号过滤器"——只有正能量通过，负能量被拦截，这避免了传统激活函数在信号过强时"饱和"的问题。Dropout 则类似于考试前的"模拟测验"——每次都随机去掉一半的神经元，剩下的神经元必须学会独立完成任务，这样在"正式考试"（推理）时所有神经元协同工作，表现更加稳健。而 GPU 并行训练就像两个人分别负责查看图像的不同区域，关键时互相交流，合作完成分类。

## 代码示例

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """简化版 AlexNet (单GPU版本)"""
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

model = AlexNet()
x = torch.randn(2, 3, 224, 224)
print(f"输出: {model(x).shape}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **突破性的性能飞跃**：AlexNet 在 ImageNet 上的压倒性胜利证明了深度卷积网络在大规模图像识别中的潜力，直接引发了计算机视觉领域的深度学习革命，开启了从手工特征（SIFT、HOG）到端到端学习的范式转变。
- **技术栈的标准化**：ReLU + Dropout + GPU 训练 + 数据增强这套组合成了此后数年深度学习训练的"标准配方"，被 VGGNet、GoogLeNet 等后续模型沿用。
- **过拟合控制方法**：AlexNet 展示了大模型配合强正则化（数据增强 + Dropout）的有效性，这一经验直接影响了后续超大模型（如 ViT）的训练策略，区别在于现代模型更多使用 LayerNorm 和权重衰减。
