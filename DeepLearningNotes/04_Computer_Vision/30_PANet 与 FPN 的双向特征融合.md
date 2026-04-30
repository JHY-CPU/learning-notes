# 30_PANet 与 FPN 的双向特征融合

## 核心概念

- **PANet（Path Aggregation Network）**：在FPN的基础上增加了一条"自底向上"的增强路径，使底层的位置信息也能高效传递到高层，形成双向特征融合。
- **FPN的单向局限**：FPN只将高层的语义信息向浅层传播（自顶向下），但浅层的位置信息无法反向传递到高层，导致高层特征缺失精细的空间定位能力。
- **自底向上增强路径**：从FPN的P2开始，步长为2的卷积下采样，逐层与FPN的对应层相加融合，生成新的特征金字塔N2→N5。
- **自适应特征池化（Adaptive Feature Pooling）**：对每个RoI，从所有金字塔层收集特征（而非仅从单一层），通过融合（相加或拼接）得到更丰富的多尺度特征表示。
- **完全连接（Fully-connected Fusion）**：掩码预测分支中，将不同层RoI的特征拼接后通过一个全连接层融合，进一步增强掩码预测的准确性。
- **PANet在多任务学习中的贡献**：PANet在COCO 2017实例分割竞赛中夺冠，同时提升了检测（框AP）和分割（掩码AP）的精度。

## 数学推导

**FPN + PANet的双向特征融合：**

FPN（自顶向下）：
$$
P_5 = \text{Conv}_{1\times1}(C_5)
$$
$$
P_4 = \text{Conv}_{1\times1}(C_4) + \text{Up}_{2\times}(P_5)
$$
$$
P_3 = \text{Conv}_{1\times1}(C_3) + \text{Up}_{2\times}(P_4)
$$
$$
P_2 = \text{Conv}_{1\times1}(C_2) + \text{Up}_{2\times}(P_3)
$$

PANet增强（自底向上）：
$$
N_2 = P_2
$$
$$
N_3 = \text{Conv}_{3\times3, s=2}(N_2) + P_3
$$
$$
N_4 = \text{Conv}_{3\times3, s=2}(N_3) + P_4
$$
$$
N_5 = \text{Conv}_{3\times3, s=2}(N_4) + P_5
$$

**自适应特征池化：**
对每个RoI，从 $N_2$ 到 $N_5$ 的所有层收集特征：
$$
\text{RoI}_{pooled} = \sum_{k=2}^5 w_k \cdot \text{RoIAlign}_k(\text{RoI}, N_k)
$$

其中 $w_k$ 是可学习的融合权重（或简单取平均）。

**PANet vs FPN的路径长度比较：**
- FPN中 $C_5 \to P_2$ 的路径涉及5次上采样和5次相加
- PANet中 $N_2 \to N_5$ 的路径涉及3次下采样和3次相加
- 底层信息到高层的路径从"穿越整个FPN再反向"缩短为"直接通过增强路径"

## 直观理解

FPN可以看作是一个"自上而下的广播系统"——高层"领导"（强语义特征）不断向基层传达指示，但基层"员工"（丰富位置信息）没有渠道向上反馈。PANet增加了一条"自下而上的反馈通道"，让基层的位置信息也能直达高层。

自适应特征池化则像是一个"多方会议"——以前每个RoI只找一个"对口负责人"（单一金字塔层）要特征，现在则让所有层的代表都参与讨论（从各层收集特征），综合各方意见后再做决策，自然比只听一面之词更全面。

## 代码示例

```python
import torch
import torch.nn as nn

class PANet(nn.Module):
    """PANet 双向特征融合（简化版）"""
    def __init__(self, in_channels_list=[256, 256, 256, 256], out_channels=256):
        super().__init__()
        # FPN（自顶向下）已在前面实现，此处接收FPN的输出[P2,P3,P4,P5]
        # 自底向上增强路径
        self.downsample_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(3)  # N2→N3, N3→N4, N4→N5
        ])
        # 融合后的输出卷积
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(4)
        ])

    def forward(self, fpn_outputs):
        # fpn_outputs: [P2, P3, P4, P5] (从浅到深)
        # 自底向上增强
        N = [fpn_outputs[0]]  # N2 = P2
        for i in range(3):
            n_down = self.downsample_convs[i](N[-1])
            n_merged = n_down + fpn_outputs[i + 1]
            N.append(n_merged)

        # 输出卷积
        outputs = []
        for i in range(4):
            outputs.append(self.output_convs[i](N[i]))
        return outputs  # [N2, N3, N4, N5]

    def adaptive_feature_pooling(self, roi_features_list):
        """自适应特征池化: 融合多层的RoI特征"""
        # roi_features_list: [N2_rois, N3_rois, N4_rois, N5_rois]
        # 每项形状: (num_rois, C, 7, 7)
        fused = torch.stack(roi_features_list, dim=0).mean(dim=0)
        return fused

# 测试
panet = PANet()
p2 = torch.randn(1, 256, 56, 56)
p3 = torch.randn(1, 256, 28, 28)
p4 = torch.randn(1, 256, 14, 14)
p5 = torch.randn(1, 256, 7, 7)

outputs = panet([p2, p3, p4, p5])
for i, out in enumerate(outputs):
    print(f"N{i+2} 尺寸: {out.shape}")
```

## 深度学习关联

- **双向特征融合的基准**：PANet提出的"自顶向下+自底向上"双向融合成为后续特征融合网络的标配。EfficientDet的BiFPN在此基础上进一步引入加权融合和跨层跳跃连接。
- **检测分割的精度提升**：PANet在COCO数据集上将Mask R-CNN的AP提升2-3个点，证明了特征融合路径设计对密集预测任务的重要性。其设计被集成到Detectron2等主流框架中。
- **NAS自动搜索特征融合**：PANet的双向设计启发了NAS-FPN（通过神经架构搜索自动设计特征金字塔拓扑），后者搜索出的不规则特征融合结构在某些场景下超过了人工设计的双向融合。
