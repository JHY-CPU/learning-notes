# 22_Faster R-CNN 与 RPN (Region Proposal Network)

## 核心概念

- **区域建议网络（RPN）**：Faster R-CNN的核心创新，用一个小型卷积网络替代Selective Search生成候选区域，实现端到端训练。RPN在卷积特征图上滑动，在每个位置预测目标边界框和目标性得分。
- **锚点（Anchor）**：在特征图每个像素位置预定义多个不同尺寸和长宽比的参考框（通常 $k=9$ 个：3种尺寸 × 3种长宽比）。RPN基于锚点预测偏移量而不是绝对坐标。
- **RPN的训练**：RPN使用二分类损失（前景/背景）+ 边界框回归损失联合训练。正样本：与任意真实框IoU>0.7的锚点；负样本：与所有真实框IoU<0.3的锚点。
- **共享特征**：RPN和检测器共享同一个卷积骨干网络的特征，避免了重复计算。这种共享是Faster R-CNN效率的关键。
- **4步交替训练**：训练RPN → 用RPN提议训练Fast R-CNN → 用RPN权重初始化并微调共享层（只调RPN）→ 微调Fast R-CNN的全连接层（固定共享层）。
- **推理流程**：RPN生成约300个候选区域 → NMS去重 → RoI Pooling提取特征 → 分类+回归。

## 数学推导

**锚点坐标映射：**
特征图上位置 $(i, j)$ 映射回原图像的中心坐标：
$$
x_{center} = i \cdot stride + offset, \quad y_{center} = j \cdot stride + offset
$$

锚点的宽高由预定义的尺寸和长宽比决定：
$$
w_k = size_{scale} \cdot \sqrt{ratio_k}, \quad h_k = size_{scale} / \sqrt{ratio_k}
$$

**RPN回归目标：**
给定锚点 $(A_x, A_y, A_w, A_h)$ 和真实框 $(G_x, G_y, G_w, G_h)$：
$$
t_x = (G_x - A_x) / A_w, \quad t_y = (G_y - A_y) / A_h
$$
$$
t_w = \log(G_w / A_w), \quad t_h = \log(G_h / A_h)
$$

**RPN的损失函数：**
$$
L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}} \sum_i L_{cls}(p_i, p_i^*) + \lambda \frac{1}{N_{reg}} \sum_i p_i^* \cdot L_{reg}(t_i, t_i^*)
$$

其中 $p_i$ 是锚点 $i$ 为目标的预测概率，$p_i^*$ 是标签（0或1）。

## 直观理解

Faster R-CNN实现了"让网络自己决定看哪里"。之前的R-CNN和Fast R-CNN依赖外部算法（Selective Search）来提议候选区域——这就像让一个专家先帮你圈出可能感兴趣的区域，你再仔细分析。Faster R-CNN把"圈区域"和"分析区域"都交给同一个网络，让视觉特征和候选区域生成互相促进。

RPN就像网络中的"注意力机制"——它在特征图的每个位置问"这里可能有物体吗？"，并用锚点作为"参考标尺"来回答"如果这里有物体，它的尺寸大概是多大？"。锚点的设计借鉴了多尺度检测的先验知识，让网络不需要从零学习物体的尺度变化。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RPN(nn.Module):
    """区域建议网络（简化版）"""
    def __init__(self, in_channels=512, mid_channels=256, anchor_sizes=[128, 256, 512],
                 aspect_ratios=[0.5, 1.0, 2.0]):
        super().__init__()
        self.num_anchors = len(anchor_sizes) * len(aspect_ratios)
        # 3x3卷积 + 两个并行的1x1卷积分支
        self.conv = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(mid_channels, self.num_anchors * 2, 1)  # 前景/背景
        self.bbox_pred = nn.Conv2d(mid_channels, self.num_anchors * 4, 1)   # 4个偏移

    def forward(self, x):
        # x: (B, C, H, W) 特征图
        x = F.relu(self.conv(x))
        objectness = self.cls_logits(x)  # (B, 2*k, H, W)
        rpn_bbox = self.bbox_pred(x)     # (B, 4*k, H, W)
        # 重排维度: 将锚点维度和空间维度展开
        N, _, H, W = objectness.shape
        objectness = objectness.view(N, -1, 2, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 2)
        rpn_bbox = rpn_bbox.view(N, -1, 4, H, W).permute(0, 3, 4, 1, 2).reshape(N, -1, 4)
        return objectness, rpn_bbox  # objectness: (N, H*W*k, 2), bbox: (N, H*W*k, 4)

class FasterRCNN(nn.Module):
    """Faster R-CNN 简化版"""
    def __init__(self, num_classes=21):
        super().__init__()
        # 骨干网络
        backbone = torchvision.models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        self.rpn = RPN(in_channels=2048)
        self.roi_pool = torchvision.ops.RoIPool((7, 7), spatial_scale=1.0/32.0)
        self.fc = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    def forward(self, x, rois=None):
        feat = self.features(x)
        objectness, rpn_bbox = self.rpn(feat)
        if rois is None:
            # 简化：从RPN输出生成候选区域（实际需要解码+NMS）
            rois = torch.tensor([[0, 0, 0, 50, 50]], dtype=torch.float32)
        x = self.roi_pool(feat, rois)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.cls_score(x), self.bbox_pred(x)

import torchvision
model = FasterRCNN()
img = torch.randn(1, 3, 224, 224)
scores, bboxes = model(img)
print(f"分类得分: {scores.shape}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **端到端可训练的检测范式**：Faster R-CNN实现了真正端到端的目标检测训练（RPN + Fast R-CNN联合优化），为后续Cascade R-CNN、Mask R-CNN等两阶段检测器奠定了基础。RPN也成为两阶段检测的标配组件。
- **锚点的普适设计模式**："预定义锚点+回归偏移"的范式被广泛用于各类检测模型——YOLOv2/v3的锚点聚类、SSD的默认框、RetinaNet的锚点等，锚点设计已成为目标检测的基础知识。
- **从两阶段到单阶段的桥梁**：RPN证明了"通过卷积网络直接生成候选区域"是可行的，这直接启发YOLO和SSD等单阶段检测器"一步到位"地预测所有边界框，不再需要显式的候选区域阶段。
