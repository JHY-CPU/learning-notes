# 54_车牌识别与场景文本检测 (DBNet)

## 核心概念

- **场景文本检测（Scene Text Detection）**：在自然场景图像中定位文字区域，输出文本的边界框（水平或旋转）。这是OCR流水线的第一步——先检测到文字在哪里，再识别是什么文字。
- **DBNet（Differentiable Binarization Network）**：Liao et al. (2020) 提出的实时场景文本检测方法，核心创新是在分割网络中加入可微分二值化模块（DB Module），将分割图转化为文本区域的二值图。
- **标准分割检测的流程**：传统分割方法先输出文本区域的概率图，再用固定阈值二值化得到文本区域，最后用后处理（如PSENet、PixelLink的连通域分析）得到文本边界框。
- **可微分二值化（DB）**：将传统二值化的固定阈值替换为可学习的阈值图，通过近似阶跃函数的可微分公式实现端到端训练。
- **自适应阈值**：DBNet不仅预测文本区域概率图，还预测每个像素的阈值图，使模型可以适应光照变化、纹理差异等复杂场景。
- **车牌识别**：将文本检测（定位车牌位置）和文本识别（读取车牌上的字符）结合的专门应用，通常使用轻量级模型实现实时识别。

## 数学推导

**可微分二值化（DB）：**

标准二值化（不可微）：
$$
B_{i,j} = \begin{cases}
1 & \text{if } P_{i,j} \ge t \\
0 & \text{otherwise}
\end{cases}
$$

可微分二值化：
$$
\hat{B}_{i,j} = \frac{1}{1 + e^{-k(P_{i,j} - T_{i,j})}}
$$

其中 $P$ 是概率图（[0,1]），$T$ 是阈值图（[0,1]），$k$ 是放大因子（通常取50）。当 $P = T$ 时，$\hat{B} = 0.5$。该公式近似阶跃函数且可微。

**DBNet的损失函数：**
$$
\mathcal{L} = \mathcal{L}_{prob} + \alpha \mathcal{L}_{threshold} + \beta \mathcal{L}_{bin}
$$

其中：
- $\mathcal{L}_{prob}$：概率图的二值交叉熵损失（正样本=文本区域）
- $\mathcal{L}_{threshold}$：阈值图的L1损失（仅在文本边界区域计算）
- $\mathcal{L}_{bin}$：近似二值图的Dice损失或BCE损失

**文本轮廓生成：**
在推理时，从二值图中提取文本轮廓的步骤：
- 对二值图进行轮廓提取（connected component）
- 对每个轮廓使用偏移量 $D = \frac{A \times r}{L}$ 进行缩放（$A$ 是轮廓面积，$L$ 是周长，$r=1.5$）
- 缩放后的多边形作为最终的文本边界框

**车牌识别中的字符分割（LPRNet）：**
车牌识别通常使用CRNN或基于注意力机制的序列识别模型。LPRNet使用轻量级CNN + RNN + CTC，特点是轻量（<1M参数）且可处理不定长的车牌字符序列。

## 直观理解

DBNet的核心创新是将"二值化阈值"从固定的超参数变成了网络可以学习的变量。传统的分割检测方法是用一个固定阈值（如0.5）来判断像素是否属于文本区域——这在均匀光照下效果不错，但遇到阴影、反光、模糊文字时，固定阈值可能太高（漏检）或太低（误检）。

DBNet同时预测"这里属于文本的概率"（概率图）和"这里的理想阈值应该是多少"（阈值图）。这样，即使某个像素的预测概率只有0.3，但如果阈值图显示这里的阈值应该很低（0.2），则这个像素仍会被判定为文本。这使DBNet对复杂场景的适应能力大大增强。

## 代码示例

```python
import torch
import torch.nn as nn

class DBHead(nn.Module):
    """DBNet 检测头 (可微分二值化)"""
    def __init__(self, in_channels=64, k=50):
        super().__init__()
        self.k = k
        # 概率图 (文本区域)
        self.prob_conv = nn.Conv2d(in_channels, 1, 3, padding=1)
        # 阈值图
        self.thresh_conv = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, x):
        prob = torch.sigmoid(self.prob_conv(x))  # 概率图 P
        thresh = torch.sigmoid(self.thresh_conv(x))  # 阈值图 T
        # 可微分二值化
        binary = torch.sigmoid(self.k * (prob - thresh))
        return prob, thresh, binary

class DBNet(nn.Module):
    """DBNet 场景文本检测 (简化版)"""
    def __init__(self, backbone_out=256):
        super().__init__()
        # 简化的特征金字塔
        self.fpn = nn.Sequential(
            nn.Conv2d(backbone_out, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.db_head = DBHead(64)
        # 文本识别分支 (简单实现)
        self.recognition = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 8)),
        )

    def forward(self, x):
        x = self.fpn(x)
        prob, thresh, binary = self.db_head(x)
        rec_feat = self.recognition(x)
        return prob, thresh, binary, rec_feat

class SimpleLicensePlateRecognition(nn.Module):
    """车牌识别模型 (基于CRNN)"""
    def __init__(self, num_chars=68, hidden=128):  # 省份缩写+字母+数字
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
        )
        self.rnn = nn.LSTM(128, hidden, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden * 2, num_chars)

    def forward(self, x):
        # x: (B, 3, H, W) 车牌图像, H通常=48
        x = self.cnn(x)  # (B, 128, H/4, W/4)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, W, C * H)  # (B, W/4, C*H/4)
        x, _ = self.rnn(x)
        x = self.fc(x)  # (B, seq_len, num_chars)
        return x

# 测试DBNet
model = DBNet(backbone_out=256)
x = torch.randn(1, 256, 64, 64)
prob, thresh, binary, rec_feat = model(x)
print(f"概率图: {prob.shape}")
print(f"阈值图: {thresh.shape}")
print(f"二值图: {binary.shape}")

# 测试车牌识别
lpr = SimpleLicensePlateRecognition(num_chars=68)
plate = torch.randn(1, 3, 48, 120)  # 常用车牌尺寸
plate_out = lpr(plate)
print(f"车牌识别输出: {plate_out.shape}")

print(f"\nDBNet参数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"LPR参数量: {sum(p.numel() for p in lpr.parameters()):,}")
```

## 深度学习关联

- **实时文本检测的SOTA**：DBNet在精度和速度之间取得了极好的平衡（在ICDAR 2015数据集上达到82% F1-score，推理速度~62FPS），成为工业OCR系统中文本检测的默认选择。后续的DBNet++进一步改进。
- **端到端OCR的演进**：从"检测+识别"两阶段到端到端OCR（如ABCNet、SPTS、MaskOCR），OCR技术正朝着更统一的框架发展。但两阶段的"检测（DBNet）+识别（CRNN）"流水线仍然是工业界最稳健的选择。
- **车牌识别的工业应用**：车牌识别（LPR）是OCR技术最成熟的工业应用之一——从停车场管理、高速公路收费到交通监控，轻量级CNN+RNN+CTC模型在嵌入式设备（如树莓派、Jetson Nano）上实现了实时高精度的车牌识别。
