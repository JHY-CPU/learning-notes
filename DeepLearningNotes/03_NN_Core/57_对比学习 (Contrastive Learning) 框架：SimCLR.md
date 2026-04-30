# 57 对比学习 (Contrastive Learning) 框架：SimCLR

## 核心概念

- **SimCLR 框架**：SimCLR（Simple Framework for Contrastive Learning of Visual Representations）是 Google 提出的简洁而有效的视觉对比学习框架。它的核心思想是最大化同一图像不同增强视图之间的一致性。

- **数据增强的关键作用**：SimCLR 使用精心设计的数据增强组合（随机裁剪+颜色抖动+高斯模糊+水平翻转），构造正样本对。增强的选择和质量直接影响学习到的表示质量——SimCLR 论文对此进行了系统性研究。

- **投影头（Projection Head）**：SimCLR 在编码器后添加一个 MLP 投影头，将表示映射到对比损失计算的空间。训练后丢弃投影头，使用编码器的输出作为通用特征表示。投影头能过滤掉增强相关的信息，保留更本质的语义。

- **NT-Xent 损失**：Normalized Temperature-Scaled Cross-Entropy Loss（NT-Xent）是 SimCLR 使用的对比损失。在 batch 内构造正负样本对，损失鼓励正样本对相似度高，负样本对相似度低。

## 数学推导

**SimCLR 的完整框架**：

给定一个 batch 的 $N$ 张图像 $\{x_k\}_{k=1}^N$：

1. 对每张图像应用两种随机增强，得到 $2N$ 个增强视图：$\tilde{x}_i$ 和 $\tilde{x}'_i$
2. 编码器 $f(\cdot)$ 提取表示：$h_i = f(\tilde{x}_i)$，$h'_i = f(\tilde{x}'_i)$
3. 投影头 $g(\cdot)$ 映射到对比空间：$z_i = g(h_i)$，$z'_i = g(h'_i)$
4. 计算 NT-Xent 损失

**NT-Xent 损失**：

对第 $i$ 个正样本对 $(z_i, z'_i)$，对比损失为：

$$
\ell(i) = -\log \frac{\exp(\text{sim}(z_i, z'_i)/\tau)}{\sum_{j=1}^{N} [\exp(\text{sim}(z_i, z'_j)/\tau) + \exp(\text{sim}(z_i, \tilde{z}_j)/\tau)]}
$$

其中 $\text{sim}(u, v) = u^T v / \|u\|\|v\|$ 是余弦相似度，$\tau$ 是温度参数。

总损失为所有正样本对的平均值：

$$
L = \frac{1}{2N} \sum_{i=1}^{N} [\ell(i) + \ell'(i)]
$$

其中 $\ell'(i)$ 是以 $z'_i$ 为锚点的对称损失。

**温度参数 $\tau$ 的作用**：

温度控制对困难负样本的关注程度：
- $\tau$ 小：对最相似的负样本施加更大惩罚（挖掘困难负样本）
- $\tau$ 大：对所有负样本"一视同仁"

在 SimCLR 中，$\tau = 0.5$ 是最优选择。

**Batch Size 的重要性**：

SimCLR 需要大 batch 来提供足够多的负样本。更大的 batch 包含更多负样本，提供更好的对比信号。SimCLR 论文使用 4096 的 batch size。

## 直观理解

SimCLR 的核心思想可以概括为"自己跟自己学"——不需要人工标注，只需要让同一张图片的不同"变体"（数据增强版本）在特征空间中靠近，同时让不同图片的特征远离。

这就像教你认识"椅子"这个概念：给你看不同角度、不同光照下的同一把椅子，让你知道这些都是"椅子"；同时给你看很多其他物体的图片，让你知道那些不是椅子。通过对比，你学会了椅子的本质特征。

为什么投影头（projection head）很重要？假设你认识一个人，无论他是穿红衣服还是蓝衣服（增强变化），你都知道是他。编码器可能记住衣服颜色这种"表面信息"，但投影头会过滤掉这些信息，只保留身份特征。训练完丢弃投影头后，编码器输出的是"去除表面信息后的本质特征"。

大 batch size 的必要性：对比学习就像"在人群中进行辨认"——你需要在大量的人（大量负样本）中认出你的朋友（正样本）。如果只有几个人（小 batch），缺乏挑战性，学不好。SimCLR 的 4096 batch 相当于在超级市场中辨认朋友。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# SimCLR 框架实现
class SimCLREncoder(nn.Module):
    """SimCLR 编码器（投影头包含在内）"""
    def __init__(self, input_dim=64, hidden_dim=256, proj_dim=128):
        super().__init__()
        # 基编码器（backbone）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # 投影头（projection head）
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x, return_projection=True):
        h = self.encoder(x)
        if return_projection:
            z = self.projector(h)
            return F.normalize(z, dim=1), h
        return h

class SimCLR:
    """SimCLR 训练器"""
    def __init__(self, encoder, temperature=0.5, device='cpu'):
        self.encoder = encoder.to(device)
        self.temperature = temperature
        self.device = device

    def nt_xent_loss(self, z1, z2):
        """计算 NT-Xent 对比损失"""
        batch_size = z1.size(0)
        # 拼接
        z = torch.cat([z1, z2], dim=0)  # (2N, D)
        # 余弦相似度矩阵
        sim = torch.mm(z, z.T) / self.temperature

        # 排除自身的掩码
        mask = torch.eye(2 * batch_size, device=self.device).bool()
        sim = sim.masked_fill(mask, -float('inf'))

        # 正样本对的索引
        pos = torch.arange(batch_size, device=self.device)
        pos_sim = torch.cat([
            sim[pos, pos + batch_size],
            sim[pos + batch_size, pos],
        ])

        # 损失
        numerator = torch.exp(pos_sim)
        denominator = torch.exp(sim).sum(dim=1)
        loss = -torch.log(numerator / denominator).mean()
        return loss

    def train_step(self, x_aug1, x_aug2, optimizer):
        """单步训练"""
        z1, _ = self.encoder(x_aug1)
        z2, _ = self.encoder(x_aug2)
        loss = self.nt_xent_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def encode(self, x):
        """获取表示（不使用投影头）"""
        with torch.no_grad():
            return self.encoder(x, return_projection=False)

# 演示 SimCLR 训练
torch.manual_seed(42)

# 创建模型和模拟数据
encoder = SimCLREncoder(64, 256, 128)
simclr = SimCLR(encoder, temperature=0.5, device='cpu')
optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

print("SimCLR 对比学习训练:")
for epoch in range(100):
    # 模拟数据：两个增强视图
    x_orig = torch.randn(64, 64)
    x_aug1 = x_orig + torch.randn_like(x_orig) * 0.15
    x_aug2 = x_orig + torch.randn_like(x_orig) * 0.15

    loss = simclr.train_step(x_aug1, x_aug2, optimizer)

    if epoch % 20 == 0:
        print(f"  Epoch {epoch:3d}, Loss: {loss:.4f}")

# 评估学习到的表示
print("\n表示质量评估:")
# 生成带类别标签的数据
torch.manual_seed(123)
X_all = torch.randn(200, 64)
y_all = (X_all[:, 0] * X_all[:, 1] > 0).long()  # 基于特征交互的类别

X_train, y_train = X_all[:50], y_all[:50]
X_test, y_test = X_all[50:], y_all[50:]

# 使用 SimCLR 学习的特征
with torch.no_grad():
    feat_train = encoder(X_train, return_projection=False)
    feat_test = encoder(X_test, return_projection=False)

# 线性分类评估
classifier = nn.Linear(256, 2)
opt = torch.optim.Adam(classifier.parameters(), lr=0.01)

for _ in range(200):
    pred = classifier(feat_train)
    loss = F.cross_entropy(pred, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

test_acc = (classifier(feat_test).argmax(1) == y_test).float().mean()
print(f"  线性分类准确率: {test_acc.item():.4f}")

# 对比：不使用 SimCLR 训练的随机特征
random_feat = torch.randn(150, 256)
random_acc = (classifier(random_feat[:150]).argmax(1) == y_test).float().mean()
print(f"  随机特征准确率: {random_acc.item():.4f}")
```

## 深度学习关联

- **视觉表示学习的里程碑**：SimCLR 是对比学习在视觉领域的里程碑工作（2020）。它证明了一个简单的对比学习框架可以在 ImageNet 上达到与监督学习相媲美的表示质量。SimCLR v2 通过更大的模型和更深的投影头进一步提升了性能。

- **负样本的必要性讨论**：SimCLR 依赖大量负样本进行对比学习，这限制了对 batch size 的需求。后续工作 BYOL（Bootstrap Your Own Latent）和 SimSiam 证明，在没有负样本的情况下也可以进行有效的对比学习（通过"自蒸馏"机制），进一步降低了对大 batch 的依赖。

- **多模态对比学习**：CLIP（Contrastive Language-Image Pre-training）将 SimCLR 的对比学习扩展到多模态场景——使用 4 亿个（图像，文本）对进行对比学习，学习到的表示可以零样本迁移到多种下游任务，展示了对比学习的巨大潜力。
