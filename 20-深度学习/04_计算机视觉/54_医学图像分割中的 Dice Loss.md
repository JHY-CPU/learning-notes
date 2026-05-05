# 55_医学图像分割中的 Dice Loss

## 核心概念

- **Dice系数（Dice Coefficient）**：衡量两个集合重叠程度的指标，定义为 $2|X \cap Y| / (|X| + |Y|)$。在医学图像分割中，$X$ 是预测的分割掩码，$Y$ 是真实掩码。
- **Dice Loss**：$1 - \text{Dice}$，用于医学图像分割的损失函数。对类别不平衡问题非常鲁棒——前景区域通常远小于背景区域，传统交叉熵损失会导致模型偏向背景。
- **与交叉熵的对比**：交叉熵逐像素独立计算损失，对所有像素一视同仁；Dice Loss从整体分割质量的角度计算损失，等价于优化mIoU指标。
- **Soft Dice**：将离散的Dice系数推广到连续概率输出，使Dice Loss可以端到端训练。$Dice = 2\sum p_i g_i / (\sum p_i^2 + \sum g_i^2)$，其中 $p_i$ 是预测概率，$g_i$ 是真实标签（0或1）。
- **平滑系数（Smooth）**：在损失中加入一个小的平滑项 $\epsilon$（如1e-5）防止分母为零，同时避免梯度爆炸。
- **多类别Dice Loss**：在多分类分割中，可以对每个类别分别计算Dice Loss后取平均，或使用广义Dice Loss（为不同类别分配不同的权重，补偿类别频率差异）。

## 数学推导

**二分类Dice系数（离散）：**
$$
\text{Dice} = \frac{2 \times |P \cap G|}{|P| + |G|} = \frac{2 \times \sum_{i=1}^N p_i g_i}{\sum_{i=1}^N p_i + \sum_{i=1}^N g_i}
$$

其中 $p_i \in \{0,1\}$ 是二元预测，$g_i \in \{0,1\}$ 是真实标签。

**Soft Dice Loss（连续概率）：**
$$
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i=1}^N p_i g_i + \epsilon}{\sum_{i=1}^N p_i + \sum_{i=1}^N g_i + \epsilon}
$$

其中 $p_i \in [0,1]$ 是模型预测概率。

**替代形式（平方和版本）：**
$$
\mathcal{L}_{Dice} = 1 - \frac{2 \sum_{i=1}^N p_i g_i + \epsilon}{\sum_{i=1}^N p_i^2 + \sum_{i=1}^N g_i^2 + \epsilon}
$$

这个版本在计算梯度时更稳定。两者都可以使用。

**广义Dice Loss（多类别）：**
$$
\mathcal{L}_{GDice} = 1 - 2 \frac{\sum_{l=1}^L w_l \sum_i p_{il} g_{il}}{\sum_{l=1}^L w_l \sum_i (p_{il} + g_{il})}
$$
$$
w_l = \frac{1}{(\sum_i g_{il})^2}
$$

其中 $w_l$ 是为类别 $l$ 分配的权重，用于补偿类别不平衡。

**Dice + 交叉熵联合损失：**
$$
\mathcal{L} = \alpha \mathcal{L}_{Dice} + (1 - \alpha) \mathcal{L}_{BCE}
$$

联合损失结合了Dice Loss的全局结构优化和交叉熵的像素级精确度，通常比单独使用任何一种效果更好。

## 直观理解

Dice Loss的设计灵感源于"评价指标即损失函数"的思想。在医学图像分割中，我们关心的评价指标是Dice系数（或IoU），但交叉熵损失和Dice系数之间存在"鸿沟"——交叉熵的最小化不等于Dice系数的最大化。

Dice Loss直接优化目标指标，而且天然解决了类别不平衡问题。一个例子：如果一张CT图像中肿瘤只占5%，在交叉熵中背景的95%像素贡献了95%的损失，模型会倾向于把所有像素都预测为背景（达到95%准确率但分割完全无效）。而Dice Loss的分子是 $2\sum p_i g_i$——只有当真正的前景像素被正确预测时，Dice才会增加。背景像素的正确预测完全不影响Dice分数。这迫使模型专注于前景区域的预测质量。

## 代码示例

```python
import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    """Soft Dice Loss (二分类)"""
    def __init__(self, smooth=1e-5, square=False):
        super().__init__()
        self.smooth = smooth
        self.square = square  # 使用平方和版本

    def forward(self, pred, target):
        # pred: (B, 1, H, W) 预测概率
        # target: (B, 1, H, W) 标签 (0或1)
        pred = pred.contiguous().view(pred.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        
        intersection = (pred * target).sum(dim=1)
        if self.square:
            union = pred.pow(2).sum(dim=1) + target.pow(2).sum(dim=1)
        else:
            union = pred.sum(dim=1) + target.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class DiceBCELoss(nn.Module):
    """Dice Loss + 二值交叉熵 联合损失"""
    def __init__(self, dice_weight=0.5, smooth=1e-5):
        super().__init__()
        self.dice_weight = dice_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, target):
        # logits: (B, 1, H, W) 未经过sigmoid
        # target: (B, 1, H, W) 标签
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, target)
        bce_loss = self.bce(logits, target)
        return self.dice_weight * dice_loss + (1 - self.dice_weight) * bce_loss

class MultiClassDiceLoss(nn.Module):
    """多类别 Dice Loss"""
    def __init__(self, num_classes, smooth=1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target):
        # pred: (B, C, H, W) softmax输出
        # target: (B, H, W) 类别索引
        target_onehot = nn.functional.one_hot(target, self.num_classes)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()
        
        dice_per_class = []
        for c in range(self.num_classes):
            p = pred[:, c]
            t = target_onehot[:, c]
            
            intersection = (p * t).sum()
            union = p.sum() + t.sum()
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_per_class.append(1 - dice)
        
        # 忽略背景类别(0)的Dice (可选)
        return torch.stack(dice_per_class).mean()

# 对比 Dice Loss vs BCE Loss 在不平衡数据上的表现
def compare_losses():
    # 模拟极度不平衡的数据 (前景只占5%)
    target = torch.zeros(4, 1, 128, 128)
    target[:, :, 60:68, 60:68] = 1.0  # 小前景区域
    
    # 两种不同的预测
    pred_all_bg = torch.sigmoid(torch.randn(4, 1, 128, 128) - 3)  # 全部预测为背景
    pred_good = target + torch.randn(4, 1, 128, 128) * 0.1  # 接近完美的预测
    
    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()
    
    print(f"全部预测为背景: BCE={bce(pred_all_bg.log(), target):.4f}, Dice={dice(pred_all_bg, target):.4f}")
    print(f"良好预测: BCE={bce(pred_good.log(), target):.4f}, Dice={dice(pred_good, target):.4f}")
    print("→ BCE下全背景预测的损失接近良好预测，但Dice下两者差距显著")
    print("→ Dice Loss 对前景预测质量的惩罚更符合分割的评价标准")

compare_losses()

# 双注意力U-Net with Dice Loss 示例
class AttentionGate(nn.Module):
    """注意力门控"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

print("Dice Loss 已成为医学图像分割的事实标准损失函数")
```

## 深度学习关联

- **医学图像分割的标准损失**：Dice Loss（及其变体Tversky Loss、Focal Tversky Loss、组合损失）是医学图像分割中最常用、最有效的损失函数，被广泛应用于UNet、nnU-Net、Attention UNet等分割网络的训练中。
- **评价指标驱动的损失设计**：Dice Loss是"评价指标即损失"理念的典型代表。这种设计思路被推广到其他任务——如目标检测中的GIoU/CIoU Loss（直接优化IoU指标）、人脸识别中的ArcFace Loss（直接优化余弦相似度）。
- **类别不平衡问题的通用解法**：Dice Loss处理类别不平衡的方法（忽略多数类，专注少数类的预测质量）被推广到其他不平衡场景——如遥感图像分割（小物体）、工业缺陷检测（小缺陷）、长尾识别等。
