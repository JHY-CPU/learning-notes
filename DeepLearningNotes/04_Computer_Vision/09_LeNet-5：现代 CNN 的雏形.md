# 09_LeNet-5：现代 CNN 的雏形

## 核心概念

- **LeNet-5 架构概述**：由 Yann LeCun 于 1998 年提出，用于手写数字识别（MNIST），是现代卷积神经网络的奠基之作。包含 7 层网络（不含输入）：2个卷积层、2个池化层、3个全连接层。
- **C1 卷积层**：$5\times5$ 卷积核 × 6，输入 $32\times32$ 灰度图，输出 $28\times28\times6$ 的特征图。
- **S2 平均池化层**：$2\times2$ 平均池化，步长2，对每个特征图下采样到 $14\times14$，同时引入可学习的缩放系数和偏置。
- **C3 卷积层**：$5\times5$ 卷积核 × 16，使用非全连接模式（每个输出通道只连接部分输入通道），减少参数量。
- **S4 平均池化层**：与S2相同，输出 $5\times5\times16$。
- **C5 卷积层**：$5\times5$ 卷积核 × 120，由于输入是 $5\times5$，等价于全连接层。
- **F6 全连接层**：将120维特征映射到84维，对应7×12的字体显示模板。
- **输出层**：使用欧几里得径向基函数（RBF）层，计算输入向量与预设模板的欧氏距离，输出10类得分。
- **训练技巧**：使用Tanh激活函数、平均池化+可学习缩放系数、非全连接卷积以打破对称性。

## 数学推导

**LeNet-5 各层尺寸变化：**

| 层 | 输入尺寸 | 操作 | 核/窗口 | 输出尺寸 |
|---|---|---|---|---|
| 输入 | $32\times32\times1$ | - | - | $32\times32\times1$ |
| C1 | $32\times32\times1$ | Conv + Tanh | $5\times5/1$, 6个 | $28\times28\times6$ |
| S2 | $28\times28\times6$ | AvgPool × 2 | $2\times2/2$ | $14\times14\times6$ |
| C3 | $14\times14\times6$ | Conv + Tanh | $5\times5/1$, 16个 | $10\times10\times16$ |
| S4 | $10\times10\times16$ | AvgPool × 2 | $2\times2/2$ | $5\times5\times16$ |
| C5 | $5\times5\times16$ | Conv + Tanh | $5\times5/1$, 120个 | $1\times1\times120$ |
| F6 | 120 | FC + Tanh | - | 84 |
| 输出 | 84 | RBF | - | 10 |

**非全连接连接表（C3层）：**
C3的16个输出通道并非连接到S2的所有6个输入通道，而是按特定模式连接：
- 前6个输出通道连接3个输入通道
- 中间6个输出通道连接4个输入通道
- 后3个输出通道连接4个输入通道
- 最后1个输出通道连接全部6个输入通道

这种设计使参数减少了约40%，同时迫使不同卷积核学习互补的特征。

## 直观理解

LeNet-5 的设计理念可以概括为"从局部到全局、从具体到抽象"。网络的早期层（C1, S2）在图像的局部区域提取低级特征（边缘、角点），中间层（C3, S4）将这些局部特征组合成稍复杂的模式（曲线、圆圈），最后全连接层（C5, F6）将这些模式抽象为完整的数字概念。这种层级式的特征提取方式被证明非常有效——人眼识别数字的过程也是类似的：先看笔画的局部走向，再综合判断是哪个数字。

## 代码示例

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    """LeNet-5 的现代实现 (使用ReLU和MaxPooling适配PyTorch)"""
    def __init__(self, num_classes=10):
        super().__init__()
        # C1: 1->6, 5x5 conv + Tanh + AvgPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # padding=2保持32x32
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # C3: 6->16, 5x5 conv + Tanh + AvgPool
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # C5: 全连接等价层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)    # 32x32 -> 16x16x6
        x = self.conv2(x)    # 16x16x6 -> 5x5x16
        x = x.view(x.size(0), -1)  # flatten
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)      # 输出10类logits
        return x

# 测试前向传播
model = LeNet5()
x = torch.randn(1, 1, 32, 32)
y = model(x)
print(f"输入尺寸: {x.shape}")
print(f"输出尺寸: {y.shape}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **现代CNN的架构模板**：LeNet-5开创的"卷积+池化+全连接"三部曲模板被后续几乎所有CNN模型（AlexNet、VGGNet、ResNet）继承。虽然现代网络更深更复杂，但核心设计逻辑与LeNet-5一脉相承。
- **层级式特征学习**：LeNet-5首先验证了神经网络可以自动学习从低级到高级的分层特征表示，这一思想直接启发了后来的迁移学习——预训练网络的前几层提取通用特征，后几层提取任务特异性特征。
- **工业OCR应用的基石**：LeNet-5在支票识别、邮政编码识别等工业场景中成功应用，证明了CNN在实际问题中的价值。今天的手写体识别、文档数字化系统仍在使用基于LeNet的设计思路。
