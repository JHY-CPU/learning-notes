# 26_SSD：多尺度特征图的目标检测

## 核心概念

- **SSD（Single Shot MultiBox Detector）**：单阶段检测器，在多个不同分辨率的特征图上直接预测边界框和类别，同时兼顾大物体和小物体的检测。
- **多尺度特征图检测**：使用CNN中不同层的特征图（从浅层到深层），浅层大特征图检测小物体，深层小特征图检测大物体。这是SSD的核心创新。
- **默认框（Default Boxes / Prior Boxes）**：在特征图每个位置预定义一组不同尺度和长宽比的先验框，类似于Faster R-CNN的锚点但更密集。
- **直接预测**：对每个默认框，直接预测其相对于默认框的偏移量和各类别置信度，不使用RPN或候选区域阶段。
- **难例挖掘（Hard Negative Mining）**：正负样本极度不平衡（负样本远多于正样本），通过选择loss最高的负样本（难例）使正负样本比例控制在1:3左右进行训练。
- **数据增强**：使用随机裁剪、颜色扭曲等多种数据增强策略（"数据增强策略"本身被视为SSD的重要贡献），显著提升小物体检测能力。

## 数学推导

**多尺度特征图的尺寸变化：**
SSD使用VGG16作为骨干网络，从6个不同层提取特征进行检测：

| 特征图 | 尺寸 | 步长 | 检测物体尺度 |
|---|---|---|---|
| Conv4_3 | $38\times38$ | 8 | 极小 |
| Conv7 (FC7) | $19\times19$ | 16 | 小 |
| Conv8_2 | $10\times10$ | 32 | 中 |
| Conv9_2 | $5\times5$ | 64 | 中大 |
| Conv10_2 | $3\times3$ | 128 | 大 |
| Conv11_2 | $1\times1$ | 256 | 极大 |

**默认框的尺度设计：**
第 $k$ 层特征图上的默认框尺寸：
$$
s_k = s_{min} + \frac{s_{max} - s_{min}}{m - 1} (k - 1), \quad k \in [1, m]
$$

其中 $s_{min}=0.2$（最浅层），$s_{max}=0.9$（最深层）。每个位置生成 $k$ 个默认框（如6个：1种尺度 + 5种长宽比）。

**SSD的损失函数：**
$$
L(x, c, l, g) = \frac{1}{N} \left(L_{conf}(x, c) + \alpha L_{loc}(x, l, g)\right)
$$

定位损失（Smooth L1）：
$$
L_{loc}(x, l, g) = \sum_{i \in Pos} \sum_{m \in \{cx,cy,w,h\}} x_{ij}^k \cdot \text{smooth}_{L1}(l_i^m - \hat{g}_j^m)
$$

置信度损失（Softmax交叉熵）：
$$
L_{conf}(x, c) = -\sum_{i \in Pos} x_{ij}^p \log(\hat{c}_i^p) - \sum_{i \in Neg} \log(\hat{c}_i^0)
$$

## 直观理解

SSD的多尺度检测思想非常直观：在CNN中，浅层特征图分辨率高、感受野小，适合检测小物体；深层特征图分辨率低、感受野大，适合检测大物体。SSD直接在不同层上"加装"检测头，让每层负责其"擅长"尺度的物体检测。

这就像一支配备了多种望远镜的观测队伍——短焦望远镜（浅层特征）视野大但看得近，负责搜寻近距离的小目标；长焦望远镜（深层特征）视野小但看得远，负责远距离的大目标。所有望远镜同时工作，互不干扰，覆盖率全面提升。

## 代码示例

```python
import torch
import torch.nn as nn

class SSD(nn.Module):
    """简化版SSD检测头"""
    def __init__(self, num_classes=21):
        super().__init__()
        # VGG16骨干
        vgg = torchvision.models.vgg16(pretrained=False)
        self.features = vgg.features[:30]  # 到Conv4_3
        
        # 额外特征层
        self.extras = nn.Sequential(
            nn.Conv2d(512, 256, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(),  # Conv8_2
            nn.Conv2d(512, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(),  # Conv9_2
            nn.Conv2d(256, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),  # Conv10_2
            nn.Conv2d(256, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),  # Conv11_2
        )

        # 每个特征图上的检测层（loc: 4*num_boxes, conf: num_classes*num_boxes）
        self.loc_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * 4, 3, padding=1),   # Conv4_3: 4个框
            nn.Conv2d(512, 6 * 4, 3, padding=1),   # Conv7: 6个框
            nn.Conv2d(512, 6 * 4, 3, padding=1),   # Conv8_2
            nn.Conv2d(256, 6 * 4, 3, padding=1),   # Conv9_2
            nn.Conv2d(256, 4 * 4, 3, padding=1),   # Conv10_2
            nn.Conv2d(256, 4 * 4, 3, padding=1),   # Conv11_2
        ])
        self.conf_layers = nn.ModuleList([
            nn.Conv2d(512, 4 * num_classes, 3, padding=1),
            nn.Conv2d(512, 6 * num_classes, 3, padding=1),
            nn.Conv2d(512, 6 * num_classes, 3, padding=1),
            nn.Conv2d(256, 6 * num_classes, 3, padding=1),
            nn.Conv2d(256, 4 * num_classes, 3, padding=1),
            nn.Conv2d(256, 4 * num_classes, 3, padding=1),
        ])

    def forward(self, x):
        # 提取特征
        sources = []
        for i in range(30):
            x = self.features[i](x)
        sources.append(x)  # Conv4_3
        # 额外层
        for i in range(0, len(self.extras), 2):
            x = self.extras[i](x)
            x = self.extras[i+1](x)
            sources.append(x)
        
        # 多尺度预测
        locs, confs = [], []
        for s, loc_layer, conf_layer in zip(sources, self.loc_layers, self.conf_layers):
            locs.append(loc_layer(s).permute(0, 2, 3, 1).contiguous())
            confs.append(conf_layer(s).permute(0, 2, 3, 1).contiguous())
        
        locs = torch.cat([l.view(l.size(0), -1) for l in locs], dim=1)
        confs = torch.cat([c.view(c.size(0), -1) for c in confs], dim=1)
        return locs, confs

import torchvision
model = SSD()
x = torch.randn(1, 3, 300, 300)
locs, confs = model(x)
print(f"定位输出: {locs.shape}, 分类输出: {confs.shape}")
print(f"默认框总数: {locs.shape[1] // 4}")
```

## 深度学习关联

- **多尺度检测的范式**：SSD开创了"在多个特征图上分别检测"的范式，直接影响了后续的MS-CNN、FPN（特征金字塔网络）、TridentNet等。FPN通过自顶向下的路径增强了各层特征，在SSD基础上进一步提升了多尺度检测质量。
- **单阶段检测的精度追赶**：SSD证明了单阶段检测器通过多尺度设计可以达到接近两阶段检测器的精度，为YOLOv2/v3的改进提供了重要参考（YOLOv3采纳了多尺度检测思想）。
- **实时检测系统的核心组件**：SSD因其在速度和精度之间的良好平衡，被广泛应用于移动端实时检测、自动驾驶中的行人/车辆检测、视频监控分析等实际场景。
