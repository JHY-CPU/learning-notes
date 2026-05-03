# 37_3D CNN 与视频动作识别 (C3D, I3D)

## 核心概念

- **3D卷积（3D Convolution）**：将标准的2D卷积扩展到时间维度，卷积核在空间（H, W）和时间（T）三个维度上滑动，同时捕获空间外观和时间运动信息。
- **C3D（Convolutional 3D）**：Tran et al. (2015) 提出的3D CNN架构，使用 $3\times3\times3$ 的3D卷积核，直接从视频帧序列中学习时空特征，常用于动作识别。
- **3D卷积 vs 2D卷积**：2D卷积在帧上独立操作，只学习空间特征；3D卷积同时在时间和空间维度上操作，能学习运动信息的时序模式。
- **I3D（Inflated 3D ConvNet）**：Carreira & Zisserman (2017) 将2D CNN（Inception V1）的参数"膨胀"到3D——将 $N\times N$ 的2D卷积核扩展为 $N\times N\times N$ 的3D卷积核，利用ImageNet预训练初始化。
- **双流I3D（Two-Stream I3D）**：使用两个I3D网络分别处理RGB帧和光流帧，最后融合预测。RGB流捕获外观信息，光流流捕获运动信息。
- **Sports-1M和Kinetics数据集**：C3D在Sports-1M上预训练，I3D在Kinetics上预训练，这些大规模视频数据集推动了视频理解研究。

## 数学推导

**3D卷积操作：**
给定输入张量 $V \in \mathbb{R}^{T \times H \times W \times C_{in}}$，3D卷积核 $K \in \mathbb{R}^{K_t \times K_h \times K_w \times C_{in} \times C_{out}}$，输出为：
$$
O_{t,h,w,c} = \sum_{\tau=0}^{K_t-1} \sum_{i=0}^{K_h-1} \sum_{j=0}^{K_w-1} \sum_{k=0}^{C_{in}-1} V_{t+\tau, h+i, w+j, k} \cdot K_{\tau, i, j, k, c}
$$

**3D卷积极大增加计算量：**
- 2D卷积（$K_h \times K_w$）计算量：$\mathcal{O}(K^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)$
- 3D卷积（$K_t \times K_h \times K_w$）计算量：$\mathcal{O}(K^3 \cdot C_{in} \cdot C_{out} \cdot T \cdot H \cdot W)$

3D卷积计算量约为2D卷积的 $K_t$ 倍。当使用 $3\times3\times3$ 核时，约为3倍。

**Inflating 2D到3D（I3D）：**
将2D卷积核 $K_{2D} \in \mathbb{R}^{K \times K}$ 膨胀为3D $K_{3D} \in \mathbb{R}^{K \times K \times K}$：
$$
K_{3D}(:, :, \tau) = \begin{cases}
\frac{1}{K_t} K_{2D} & \text{每个时间切片} \\
0 & \text{如果不需要时间建模}
\end{cases}
$$

**池化层的膨胀：**
2D MaxPool $2\times2$（步长2）→ 3D MaxPool $1\times2\times2$（不池化时间维）或 $2\times2\times2$（池化所有维）。

## 直观理解

3D卷积像是在看视频时"扫描"时空立方体。2D卷积每次只看一张图片的一个小区域（空间感受野），而3D卷积同时观看连续几帧的一个小区域（时空感受野）。这样，3D卷积不仅能识别"这是一个球"（空间特征），还能识别"球正在移动"（时间特征）。

可以把3D卷积想象成把视频切成很多"时空小块"（$3\times3\times3$ 的立方体），每个小块既是三维空间中的一个局部区域，也包含了连续3帧的信息。网络通过这些时空小块来学习动作模式——比如"挥手"在时空上会形成一个扇形的激活模式。

I3D的关键创新是"膨胀"：先将一个在ImageNet上预训练好的2D网络（Inception V1）的卷积核在时间维上"拉伸"，再用视频数据微调。这让3D网络"继承"了2D网络学到的强大空间特征，减少了对海量视频标注数据的依赖。

## 代码示例

```python
import torch
import torch.nn as nn

class C3D(nn.Module):
    """C3D 网络 (简化版)"""
    def __init__(self, num_classes=101):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.fc6 = nn.Linear(512, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: (B, 3, T, H, W)
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.pool4(x)
        x = self.relu(self.conv5a(x))
        x = self.relu(self.conv5b(x))
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc6(x)))
        x = self.dropout(self.relu(self.fc7(x)))
        x = self.fc8(x)
        return x

# 3D vs 2D 卷积参数量对比
conv2d = nn.Conv2d(64, 128, kernel_size=3, padding=1)
conv3d = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
print(f"2D卷积参数量: {sum(p.numel() for p in conv2d.parameters())}")
print(f"3D卷积参数量: {sum(p.numel() for p in conv3d.parameters())}")

# 测试C3D
model = C3D(num_classes=101)
x = torch.randn(1, 3, 16, 112, 112)  # 16帧, 112x112
out = model(x)
print(f"C3D输出: {out.shape}")
print(f"C3D参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **视频理解的基础架构**：C3D和I3D奠定了3D CNN在视频理解领域的基础地位，后续的SlowFast网络（双路径不同帧率）、Non-local Network（非局部注意力）、Video Swin Transformer等都是在3D时空建模基础上的改进。
- **预训练策略的演变**：从C3D在Sports-1M预训练、I3D的"膨胀初始化"，到后来SlowFast在Kinetics大规模预训练，视频模型的预训练数据规模和模型容量都在持续增长。
- **从3D CNN到视频Transformer**：近年来，Video Vision Transformer (ViViT)、TimeSformer、Video Swin Transformer等将自注意力引入视频理解，通过"时空自注意力"替代3D卷积，在多个视频理解基准上超越了传统3D CNN方法。
