# 31_图像分类中的标签平滑与 Mixup 增强

## 核心概念

- **标签平滑（Label Smoothing）**：将one-hot硬标签（如 $[0,0,1,0]$）替换为软标签（如 $[0.025,0.025,0.925,0.025]$），即在真实类别上分配 $1-\epsilon$ 的置信度，其余 $\epsilon$ 均匀分配给其他类别。$\epsilon$ 通常取 0.1。
- **标签平滑的动机**：防止模型过于自信（over-confidence），缓解过拟合。one-hot标签鼓励模型输出logit趋向无穷大，导致过拟合和校准不良。
- **Mixup数据增强**：将两张随机选取的图像及其标签按比例混合：$\tilde{x} = \lambda x_i + (1-\lambda) x_j$，$\tilde{y} = \lambda y_i + (1-\lambda) y_j$，其中 $\lambda \sim \text{Beta}(\alpha, \alpha)$。
- **Mixup的线性行为**：Mixup强制模型在训练样本之间学习简单的线性行为，增强模型的泛化能力和对对抗样本的鲁棒性。
- **标签平滑与知识蒸馏的关系**：标签平滑可以看作是一种"静态的"知识蒸馏——用均匀分布作为"教师"的软标签来指导学生模型。
- **实现细节**：标签平滑和Mixup通常在训练时使用，推理时不做任何修改。两者可以组合使用，进一步提升性能。

## 数学推导

**标签平滑的公式：**
$$
y^{LS}_k = y_k (1 - \epsilon) + \frac{\epsilon}{K}
$$

其中 $y_k$ 是one-hot标签（第 $k$ 类为1，其余为0），$K$ 是类别总数，$\epsilon$ 是平滑系数。

**标签平滑后的交叉熵损失：**
$$
\mathcal{L}_{LS} = -\sum_{k=1}^K y^{LS}_k \log(p_k)
$$
$$
= -\sum_{k=1}^K [y_k(1-\epsilon) + \frac{\epsilon}{K}] \log(p_k)
$$
$$
= (1-\epsilon) \mathcal{L}_{CE} + \frac{\epsilon}{K} \sum_{k=1}^K \log(p_k)
$$

第一项是标准交叉熵，第二项是均匀分布的KL散度，鼓励所有类别的预测概率接近 $1/K$。

**Mixup的损失函数：**
$$
\mathcal{L}_{Mixup} = \lambda \cdot \mathcal{L}_{CE}(f(\tilde{x}), y_i) + (1-\lambda) \cdot \mathcal{L}_{CE}(f(\tilde{x}), y_j)
$$

其中 $f(\tilde{x})$ 是模型对混合图像 $\tilde{x}$ 的预测输出。

## 直观理解

标签平滑可以理解为"让模型谦逊一点"。one-hot标签相当于告诉模型"这就是猫，100%确定"，模型为了迎合这种绝对标签，会将猫类的logit推得非常大，产生过度自信的决策边界。标签平滑则告诉模型"这有92.5%的概率是猫，但也有可能是狗或鸟"，让模型的决策边界更柔和，泛化更好。

Mixup可以理解为"在数据之间搭桥"。传统数据增强（旋转、裁剪等）只在单个样本的附近探索，而Mixup在两个样本之间的连线上创造新的训练数据。这就像在已知的城市之间修建高速公路——模型不仅知道A点和B点，还知道从A到B的整条路径上的情况，决策边界更加平滑。

## 代码示例

```python
import torch
import torch.nn as nn
import numpy as np

class LabelSmoothingCELoss(nn.Module):
    """带标签平滑的交叉熵损失"""
    def __init__(self, num_classes, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred: (N, C) logits
        # target: (N,) hard labels
        log_probs = nn.functional.log_softmax(pred, dim=-1)
        # 构造平滑标签
        with torch.no_grad():
            smooth_labels = torch.full_like(log_probs, self.smoothing / self.num_classes)
            smooth_labels.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
        loss = -(smooth_labels * log_probs).sum(dim=1).mean()
        return loss

def mixup_data(x, y, alpha=1.0):
    """Mixup 数据增强"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = (y, y[index])  # 返回两个标签和lambda
    return mixed_x, mixed_y, lam

# 测试 Mixup + Label Smoothing 的损失计算
criterion = LabelSmoothingCELoss(num_classes=10, smoothing=0.1)
logits = torch.randn(4, 10)  # 模拟4个样本的logits
targets = torch.tensor([0, 2, 1, 3])
loss = criterion(logits, targets)
print(f"标签平滑损失: {loss.item():.4f}")

# 测试 Mixup
x = torch.randn(4, 3, 32, 32)
y = torch.tensor([0, 1, 2, 3])
mixed_x, (y_a, y_b), lam = mixup_data(x, y, alpha=1.0)
print(f"Mixup 后图像形状: {mixed_x.shape}, lambda={lam:.3f}")

# Mixup + 标签平滑联合损失
criterion_mixup = LabelSmoothingCELoss(num_classes=10, smoothing=0.1)
logits_mixed = torch.randn(4, 10)  # 模拟模型对混合图像的输出
loss_mixup = lam * criterion_mixup(logits_mixed, y_a) + \
             (1 - lam) * criterion_mixup(logits_mixed, y_b)
print(f"Mixup + 标签平滑损失: {loss_mixup.item():.4f}")
```

## 深度学习关联

- **训练技巧的标准组合**：标签平滑和Mixup已成为ImageNet分类训练的"标准套餐"，广泛用于ResNet、EfficientNet、ViT等模型的训练流程中。特别是ViT，由于其缺乏CNN的归纳偏置，对标签平滑和Mixup的依赖更大。
- **模型校准（Model Calibration）**：标签平滑是现代神经网络校准的重要工具，能够使模型的置信度更接近真实准确率。这对高风险应用（医疗、自动驾驶）中模型的可靠决策至关重要。
- **数据增强的理论理解**：Mixup启发了一系列"混合"类数据增强方法——CutMix（区域级混合）、Manifold Mixup（特征空间混合）、FMix（傅里叶域混合）等，推动了数据增强从"经验技巧"向"理论指导"的方向发展。
