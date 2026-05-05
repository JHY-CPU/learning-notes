# 03_GAN 训练不稳定性与模式崩溃 (Mode Collapse)

## 核心概念

- **模式崩溃 (Mode Collapse)**：生成器只学习到了真实分布中的少数模式，反复生成相似的样本，丧失了多样性。
- **训练不稳定性**：GAN 的对抗训练过程容易震荡、发散，难以收敛到纳什均衡。D 和 G 的损失曲线通常剧烈波动。
- **梯度消失**：当判别器过于强大时，生成器的梯度趋于零，无法继续学习。这在使用原始 GAN 损失时尤为常见。
- **模式坍塌的类型**：完全坍塌（只生成一种样本）、周期性坍塌（在不同模式间切换）、部分坍塌（多样性不足但非完全坍塌）。
- **梯度惩罚与谱归一化**：通过对判别器的 Lipschitz 连续性施加约束来稳定训练，代表方法有 WGAN-GP 和 Spectral Norm。
- **多样性与质量权衡**：改善模式崩溃有时会降低单样本质量，反之亦然——这是 GAN 训练中的核心矛盾。

## 数学推导

**原始 GAN 损失与梯度消失**：

判别器最优时，生成器的损失为：

$$
C(G) = -\log 4 + 2 \cdot JS(p_{\text{data}} \parallel p_g)
$$

当 $p_{\text{data}}$ 与 $p_g$ 的支撑集不重叠时（高维空间中几乎必然发生），JS 散度为常数 $\log 2$，梯度为零。这就是**梯度消失**的数学本质。

**模式崩溃的条件**：当生成器发现欺骗判别器的最优策略是专注于少数高概率模式时，就会发生模式崩溃。数学上，生成器在优化：

$$
\min_G \mathbb{E}_{z \sim p_z} [f(D(G(z)))]
$$

如果某些模式 $x_i$ 对应的 $D(x_i)$ 容易逼近 1，G 就会倾向于只生成这些模式。

**WGAN 对模式崩溃的缓解**：WGAN 使用 Wasserstein 距离替代 JS 散度：

$$
W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]
$$

即使在支撑集不重叠时，Wasserstein 距离仍然提供有意义的梯度，因此能有效缓解模式崩溃。

## 直观理解

- **模式崩溃**就像学生只做会做的几道题来应付考试，遇到没见过的题就放弃——他的"知识分布"只覆盖了真实分布的一小部分。
- **梯度消失**像是警察太强了，造假者一出手就被抓，造假者得不到任何反馈来改进技术，最终放弃造假（梯度为零）。
- **周期性坍塌**则像是造假者学会了造美元，警察就专门查美元；造假者转向造欧元，警察又专查欧元——如此循环往复。
- **不稳定性**的根源在于 G 和 D 在玩一个"猫鼠游戏"，双方都在移动靶上优化，自然难以稳定收敛。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 检测模式崩溃的辅助函数
def compute_diversity(images, threshold=0.01):
    """计算生成图像的多样性：成对平均距离"""
    batch_size = images.size(0)
    flat = images.view(batch_size, -1)
    dists = torch.cdist(flat, flat)
    mean_dist = dists[~torch.eye(batch_size, dtype=torch.bool)].mean()
    return mean_dist.item()

# 谱归一化（Spectral Normalization）层
def spectral_norm(module):
    """对线性层或卷积层应用谱归一化"""
    return nn.utils.spectral_norm(module)

class StableDiscriminator(nn.Module):
    """使用谱归一化的稳定判别器"""
    def __init__(self, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(data_dim, 256)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 1))
        )
    
    def forward(self, x):
        return self.net(x)

# 特征匹配（Feature Matching）：防止模式崩溃的技巧
class GeneratorWithFM(nn.Module):
    """带特征匹配的生成器"""
    def __init__(self, latent_dim=100, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, data_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

def feature_matching_loss(G, D, real_features, z):
    """生成器的特征匹配损失：匹配判别器中间层特征"""
    fake = G(z)
    fake_features = D.net[:2](fake)  # 取判别器中间层
    return torch.mean((real_features - fake_features) ** 2)

# 训练时监控模式崩溃
latent_dim = 100
G = GeneratorWithFM(latent_dim)
D = StableDiscriminator()

# mini-batch discrimination（另一种抗模式崩溃技术）：为每个样本添加唯一编码
def minibatch_stddev(x, group_size=4):
    """Mini-batch 标准差层：增加批内多样性"""
    std = x.std(dim=0, keepdim=True).mean()
    std_feat = std.expand(x.size(0), 1)
    return torch.cat([x, std_feat], dim=1)

print(f"生成器参数量: {sum(p.numel() for p in G.parameters())}")
print(f"判别器参数量: {sum(p.numel() for p in D.parameters())}")
print("谱归一化 + 特征匹配配置完成")
```

## 深度学习关联

- **WGAN-GP**：用 Wasserstein 距离加梯度惩罚替代原始 GAN 损失，从根本上改善了训练稳定性，是 GAN 训练的重要里程碑。
- **StyleGAN 系列**：通过渐进式训练、风格调制、路径长度正则等技术，在保持多样性的同时大幅提升了生成质量。
- **扩散模型的崛起**：由于 GAN 训练不稳定性难以根本解决，学界和工业界逐渐转向训练更稳定的扩散模型。
- **实际工程技巧**：标签平滑（Label Smoothing）、小批量判别（Mini-batch Discrimination）、历史平均（Historical Averaging）等技巧在实践中广泛用于稳定 GAN 训练。
