# 02_GAN 基础：生成器与判别器的博弈论视角

## 核心概念
- **零和博弈**：生成器（Generator, G）与判别器（Discriminator, D）构成一个二人零和博弈，G 的收益等于 D 的损失，反之亦然。
- **纳什均衡**：GAN 训练的终极目标是达到纳什均衡——此时 G 生成的数据完全无法被 D 区分，D 只能以 1/2 概率猜测。
- **生成器**：接收随机噪声向量 $z \sim p_z(z)$，通过神经网络映射到数据空间，目标是生成逼真样本以"欺骗"判别器。
- **判别器**：接收真实数据或生成数据，输出一个标量表示样本为真的概率，目标是准确区分真实与生成样本。
- **极小极大博弈**：GAN 的价值函数 $V(D,G)$ 定义为 $\min_G \max_D V(D,G)$，G 试图最小化而 D 试图最大化。
- **训练动态**：交替训练——先更新判别器 K 步，再更新生成器 1 步，形成循环对抗训练过程。

## 数学推导

**原始 GAN 价值函数**：

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]
$$

**判别器的最优解**：固定 G，最优 D 为

$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}
$$

代入 $D^*$ 可得 G 的损失函数等价于最小化 $p_{\text{data}}$ 与 $p_g$ 之间的 Jensen-Shannon 散度：

$$
C(G) = -\log 4 + 2 \cdot JS(p_{\text{data}} \parallel p_g)
$$

其中 JS 散度定义为：

$$
JS(p \parallel q) = \frac{1}{2}KL(p \parallel \frac{p+q}{2}) + \frac{1}{2}KL(q \parallel \frac{p+q}{2})
$$

**非饱和损失（Practical Loss）**：实践中为避免梯度消失，生成器通常最大化 $\log D(G(z))$ 而非最小化 $\log(1 - D(G(z)))$。

## 直观理解
- **造假者与警察**：生成器是造假币的，判别器是验钞的警察。造假者不断改进技术，警察不断提升鉴别能力。最终造假者造出的假币真到警察也无法分辨。
- **博弈的收敛**：理想情况下，这个博弈会收敛到纳什均衡——此时判别器只能随机猜测，而生成器掌握了真实数据的完美分布。
- **为什么有效**：GAN 巧妙地将"生成"问题转化为"对抗"问题，避免了直接最大化似然所需的显式概率建模（如 VAE 需要计算 ELBO）。
- **训练的不稳定性**：零和博弈的梯度下降不保证收敛，D 和 G 的平衡极其脆弱，这就是为什么 GAN 训练以困难著称。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 生成器：从噪声映射到数据
class Generator(nn.Module):
    def __init__(self, latent_dim=100, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, data_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

# 判别器：二分类器
class Discriminator(nn.Module):
    def __init__(self, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

# 训练循环核心逻辑
def train_gan_step(G, D, opt_G, opt_D, real_data, latent_dim):
    batch_size = real_data.size(0)
    
    # 1. 训练判别器：最大化 log D(x) + log(1 - D(G(z)))
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    
    d_real = D(real_data)
    d_fake = D(fake_data.detach())
    
    loss_D = -torch.mean(torch.log(d_real + 1e-8) + torch.log(1 - d_fake + 1e-8))
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()
    
    # 2. 训练生成器：最大化 log D(G(z))（非饱和损失）
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    d_fake = D(fake_data)
    
    loss_G = -torch.mean(torch.log(d_fake + 1e-8))
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    
    return loss_D.item(), loss_G.item()

# 初始化
latent_dim = 100
G = Generator(latent_dim)
D = Discriminator()
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
print("GAN 模型初始化完成")
```

## 深度学习关联
- **对抗训练（Adversarial Training）**：GAN 的博弈思想启发了对抗训练（Adversarial Robustness），即生成对抗样本来提升模型鲁棒性。
- **对比学习**：SimGAN 等模型将 GAN 的判别思想用于无监督表征学习，用生成器做数据增强，用判别器学习不变性特征。
- **多模态生成**：StackGAN、AttnGAN 等扩展了 GAN 到文本到图像生成，条件GAN（cGAN）引入条件信息控制生成内容。
- **前沿替代**：虽然扩散模型在图像质量上超越了 GAN，但 GAN 在推理速度（单步生成）和某些场景（如高分辨率图像翻译）中仍有不可替代的优势。
