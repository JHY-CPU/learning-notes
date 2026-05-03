# 56_自监督学习 (Self-Supervised Learning) 概述

## 核心概念

- **自监督学习的定义**：自监督学习（Self-Supervised Learning, SSL）是一种无需人工标注的表示学习方法。它利用数据本身的结构设计"前置任务"（pretext task），从无标签数据中自动生成监督信号进行训练。
- **前置任务（Pretext Task）**：SSL 的核心是设计巧妙的代理任务，使得学习到的表示能够捕获数据的有用语义特征。常见的前置任务包括：图像修复（inpainting）、旋转预测（rotation prediction）、上下文相似性（context similarity）、对比学习（contrastive learning）等。
- **对比学习 vs 生成式 SSL**：对比学习通过区分正负样本对学习表示（如 SimCLR、MoCo），生成式 SSL 通过重建输入数据学习表示（如 Masked Autoencoders、BERT 的 masked language modeling）。
- **与监督学习的对比**：SSL 不依赖人工标签，可以在海量无标签数据上训练（如 ImageNet 的 14M 张无标签图像、互联网的 TB 级文本）。学习到的表示可以迁移到下游任务，在标注数据有限时特别有价值。

## 数学推导

**对比学习的通用框架**：

给定一批数据，SSL 的目标是最大化正样本对的相似度，最小化负样本对的相似度。

InfoNCE 损失：

$$
L_{\text{InfoNCE}} = -\mathbb{E}_{x, x^+, \{x^-\}} \left[ \log \frac{\exp(f(x)^T f(x^+)/\tau)}{\exp(f(x)^T f(x^+)/\tau) + \sum_{i=1}^{N} \exp(f(x)^T f(x_i^-)/\tau)} \right]
$$

其中 $f$ 是编码器，$\tau$ 是温度参数。

**掩码自编码器（MAE）**：

输入图像被分成 patches，大部分 patches 被随机掩码。模型只处理可见 patches，目标是从可见部分重建被掩码的像素。

损失：在掩码区域上的 MSE 损失：

$$
L_{\text{MAE}} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \|x_i - \hat{x}_i\|^2
$$

其中 $\mathcal{M}$ 是被掩码的 patch 集合。

**旋转预测**：

对输入图像施加 4 种旋转（0°、90°、180°、270°），模型预测旋转类别：

$$
L_{\text{RotNet}} = -\sum_{i=1}^{4} \delta_{r_i, \hat{r}} \log p(r_i | x)
$$

## 直观理解

自监督学习可以类比为"自学成才"——人类可以在没有老师指导的情况下，通过观察世界学习知识。比如，看到一个物体的不同角度和变形，人类能学会识别这个物体，而不需要别人告诉你"这叫椅子"。

前置任务就像是"从无处不在的免费数据中挖掘监督信号"：
- **图像修复**：遮盖图片的一部分，让模型预测被遮盖的部分。模型需要理解图片的整体语义才能准确预测。
- **旋转预测**：判断图片是否被旋转了。模型需要识别物体正确的朝向，这要求理解物体的概念。
- **对比学习**：判断两张图片是否是同一物体的不同变形。模型需要理解"不变性"——同一物体在不同条件下的共同本质。

对比学习中的正负样本对：正样本对是同一图像的不同增强，负样本对来自不同图像。这迫使模型学到"对数据增强不变"的特征——这些特征恰好捕捉了语义信息。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 简单的自监督对比学习框架
class SimCLRExample(nn.Module):
    """简化版 SimCLR 对比学习"""
    def __init__(self, input_dim=64, proj_dim=128):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # 投影头（projection head）
        self.projector = nn.Sequential(
            nn.Linear(64, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=1)

def contrastive_loss(z1, z2, temperature=0.5):
    """NT-Xent 损失（对比学习损失）"""
    batch_size = z1.size(0)
    # 拼接正样本对
    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    # 计算相似度矩阵
    sim = torch.mm(z, z.T) / temperature  # (2B, 2B)

    # 掩码：排除自身
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim = sim.masked_fill(mask, -float('inf'))

    # 正样本对索引：i 与 i+B 是一对
    pos_indices = torch.arange(batch_size, device=z.device)
    pos = torch.cat([
        sim[pos_indices, pos_indices + batch_size],  # z1 与 z2
        sim[pos_indices + batch_size, pos_indices],  # z2 与 z1
    ])

    # 对比损失
    numerator = torch.exp(pos)
    denominator = torch.exp(sim).sum(dim=1)
    loss = -torch.log(numerator / denominator).mean()
    return loss

# 演示自监督预训练
torch.manual_seed(42)

model = SimCLRExample(64, 128)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("自监督对比学习预训练:")
for epoch in range(100):
    # 生成模拟数据：两个增强视图
    x = torch.randn(32, 64)  # 原始数据
    # 模拟两种不同的数据增强
    x_aug1 = x + torch.randn_like(x) * 0.1  # 增强视图 1
    x_aug2 = x + torch.randn_like(x) * 0.1  # 增强视图 2

    z1 = model(x_aug1)
    z2 = model(x_aug2)

    loss = contrastive_loss(z1, z2)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}, Contrastive Loss: {loss.item():.4f}")

# 下游分类任务评估
print("\n下游分类评估:")
# 生成带标签的数据（少量）
X_train = torch.randn(20, 64)
y_train = (X_train.sum(1) > 0).long()
X_test = torch.randn(30, 64)
y_test = (X_test.sum(1) > 0).long()

# 使用自监督预训练的特征
with torch.no_grad():
    feat_train = model.encoder(X_train).detach()
    feat_test = model.encoder(X_test).detach()

# 训练简单的线性分类器
classifier = nn.Linear(64, 2)
opt_cls = torch.optim.Adam(classifier.parameters(), lr=0.01)

for _ in range(100):
    pred = classifier(feat_train)
    loss = F.cross_entropy(pred, y_train)
    opt_cls.zero_grad()
    loss.backward()
    opt_cls.step()

with torch.no_grad():
    test_pred = classifier(feat_test).argmax(1)
    acc = (test_pred == y_test).float().mean()
    print(f"  线性分类准确率: {acc.item():.4f}")

# 对比：使用随机特征
feat_random = torch.randn(30, 64)
random_pred = classifier(feat_random).argmax(1)
random_acc = (random_pred == y_test).float().mean()
print(f"  随机特征准确率: {random_acc.item():.4f}")
```

## 深度学习关联

- **视觉领域的突破**：自监督学习在计算机视觉领域取得了革命性进展。SimCLR、MoCo、BYOL、DINO 等对比学习方法在 ImageNet 线性探测（linear probing）上已接近监督学习的性能。掩码自编码器（MAE, Masked Autoencoder）在 ViT 上取得了出色的迁移学习效果。
- **NLP 的基础**：自监督学习是 NLP 领域的基础训练范式。BERT 的掩码语言建模（Masked Language Modeling）、GPT 的自回归语言建模（Autoregressive Language Modeling）都是自监督学习。这些模型在海量无标签文本上预训练，然后适配到各种下游任务。
- **多模态自监督学习**：自监督学习正在向多模态方向扩展。CLIP（Contrastive Language-Image Pre-training）使用 4 亿个图像-文本对进行对比学习，学习到的多模态表示可以零样本迁移到多种视觉任务。这代表了自监督学习从单模态到多模态的重大演进。
