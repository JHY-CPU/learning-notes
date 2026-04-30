# 20_R-CNN：区域选择与特征提取的两阶段逻辑

## 核心概念

- **两阶段检测范式**：R-CNN（Region-based CNN）开创了"先提候选区域，再对每个区域分类"的两阶段目标检测框架，是Faster R-CNN、Mask R-CNN等系列模型的基础。
- **选择性搜索（Selective Search）**：传统方法生成候选区域（Region Proposals），通过合并相似的颜色、纹理、大小和形状的相邻区域，生成约2000个可能包含物体的候选框。
- **区域变形（Warping）**：由于候选区域大小不一，而全连接层需要固定输入尺寸，R-CNN将每个候选区域变形（warp）到 $227\times227$ 的统一尺寸。
- **SVM分类器**：对每个类别训练一个二分类SVM（而非直接使用softmax），判断候选区域是否包含该类物体。SVM在后深度学习时代仍被用作强分类器。
- **边界框回归（Bounding Box Regression）**：在分类后，对每个正样本候选框进一步回归微调其位置，输出更精确的边界框坐标 $[x, y, w, h]$。
- **训练分阶段**：R-CNN需要分步骤训练——先微调CNN（ImageNet预训练），再训练SVM分类器，最后训练边界框回归器。流程复杂且速度慢（每张图需处理2000个候选框）。

## 数学推导

**选择性搜索的相似度融合：**
四种相似度度量：
$$
s_{color}(r_i, r_j) = \sum_{k=1}^n \min(c_i^k, c_j^k) \quad (\text{颜色直方图重叠})
$$
$$
s_{texture}(r_i, r_j) = \sum_{k=1}^n \min(t_i^k, t_j^k) \quad (\text{纹理直方图重叠})
$$
$$
s_{size}(r_i, r_j) = 1 - \frac{size(r_i) + size(r_j)}{size(img)} \quad (\text{促进小区域合并})
$$
$$
s_{fill}(r_i, r_j) = 1 - \frac{size(BB_{ij}) - size(r_i) - size(r_j)}{size(img)} \quad (\text{形状兼容性})
$$

最终相似度：$s(r_i, r_j) = a_1 s_{color} + a_2 s_{texture} + a_3 s_{size} + a_4 s_{fill}$

**边界框回归的损失函数：**
给定预测框 $P = (P_x, P_y, P_w, P_h)$ 和真实框 $G = (G_x, G_y, G_w, G_h)$，回归目标是学习变换 $t$：
$$
t_x = (G_x - P_x) / P_w, \quad t_y = (G_y - P_y) / P_h
$$
$$
t_w = \log(G_w / P_w), \quad t_h = \log(G_h / P_h)
$$

损失函数为：$L_{reg} = \sum_{i \in \{x,y,w,h\}} (t_i - \hat{t}_i)^2$

**NMS（非极大值抑制）：**
对所有IoU超过阈值（如0.5）且属于同一类别的检测框，保留得分最高的框，抑制其他框。

## 直观理解

R-CNN的"两阶段"逻辑可以类比为一个"先粗筛后细筛"的应聘流程。第一阶段（选择性搜索）好比HR部门从海量简历中初步筛选出2000个"可能符合要求"的候选人（候选区域），不要求特别精确，但要把潜在人选都覆盖到。第二阶段（CNN + SVM）好比技术主管对这些候选人进行深入评估——提取简历的关键信息（CNN特征），判断是否符合岗位要求（SVM分类），对基本符合的进一步给出"推荐等级"（边界框回归）。

核心短板在于效率——每个候选人（候选框）都需要独立进行完整的面试（CNN前向），2000个候选框共享大量重叠区域却无法复用计算。

## 代码示例

```python
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms

class RCNN(nn.Module):
    """简化 R-CNN：AlexNet特征提取 + 分类/回归"""
    def __init__(self, num_classes=20):
        super().__init__()
        # 使用预训练AlexNet作为特征提取器
        alexnet = torchvision.models.alexnet(pretrained=False)
        # 去掉最后的分类层，保留到pool5 (features + avgpool)
        self.features = alexnet.features
        self.pool5 = nn.AdaptiveAvgPool2d((6, 6))
        # 全连接特征层
        self.fc6 = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        # 分类器 (使用SVM的替代方案：用softmax分类)
        self.classifier = nn.Linear(4096, num_classes)
        # 边界框回归器
        self.bbox_regressor = nn.Linear(4096, num_classes * 4)

    def forward(self, x):
        # x: (B, 3, 227, 227) 已经变形后的候选区域
        x = self.features(x)
        x = self.pool5(x)
        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        scores = self.classifier(x)
        bboxes = self.bbox_regressor(x)
        return scores, bboxes

# 模拟推理
model = RCNN(num_classes=20)
# 模拟一个候选区域经过warping后
region = torch.randn(1, 3, 227, 227)
scores, bboxes = model(region)
print(f"分类得分: {scores.shape}")  # (1, 20)
print(f"边界框回归: {bboxes.shape}")  # (1, 80) = 20类 * 4个坐标

# 总参数量
print(f"R-CNN参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **R-CNN系列的开端**：R-CNN开创的两阶段检测范式成为后续改进的基础。Fast R-CNN引入RoI Pooling实现单次特征提取共享，Faster R-CNN用RPN替代Selective Search实现端到端训练，但核心的"proposal + classification"两阶段逻辑保持不变。
- **CNN特征作为通用视觉表示**：R-CNN首次证明了在ImageNet上预训练的CNN特征可以有效地迁移到目标检测任务中（即使目标域与ImageNet差异很大），为迁移学习提供了重要实践证据。
- **候选区域方法的影响**：虽然两阶段检测逐渐被单阶段检测（YOLO、SSD）挑战效率和速度，但在需要高精度的场景（如医疗影像、工业检测）中，两阶段方法仍然因其较高的检测精度而被广泛采用。
