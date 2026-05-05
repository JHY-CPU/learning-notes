# 10_VAE 变分自编码器原理

## 核心概念

- **变分自编码器 (VAE)**：一种深度生成模型，结合了自编码器的架构和变分推断的数学框架，能学习数据的潜在表示并生成新样本。
- **编码器-解码器架构**：编码器 $q_\phi(z|x)$ 将输入 $x$ 映射到潜变量 $z$ 的分布（均值和方差），解码器 $p_\theta(x|z)$ 从潜变量重建数据。
- **变分下界 (ELBO)**：VAE 的核心优化目标——证据下界（Evidence Lower Bound），最大化 ELBO 等价于最大化数据的对数似然 $\log p(x)$。
- **概率潜空间**：不同于普通自编码器的确定性潜码，VAE 的潜空间是一个概率分布（通常是高斯分布），使得潜空间连续且有结构。
- **KL 散度正则化**：鼓励后验分布 $q_\phi(z|x)$ 接近先验分布 $p(z)$（通常是标准正态分布），防止潜空间过于分散。
- **生成新样本**：训练完成后，直接从先验 $p(z)$ 采样 $z$，通过解码器生成全新的数据。

## 数学推导

**VAE 的目标：最大化数据对数似然的 ELBO**：

$$
\log p(x) \geq \mathcal{L}(x; \theta, \phi) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) \parallel p(z))
$$

ELBO 的两项含义：
- **重建项 (Reconstruction Term)**：$\mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)]$ —— 解码器从潜码重建数据的质量
- **KL 正则项 (KL Divergence)**：$KL(q_\phi(z|x) \parallel p(z))$ —— 后验分布与先验分布的偏离程度

**完整推导**：

从贝叶斯出发：$\log p(x) = KL(q(z|x) \parallel p(z|x)) + \mathcal{L}(x)$

由于 KL 散度非负，$\log p(x) \geq \mathcal{L}(x)$，因此称为"下界"。

展开 ELBO：

$$
\mathcal{L} = \int q_\phi(z|x) \log \frac{p_\theta(x|z)p(z)}{q_\phi(z|x)} dz = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) \parallel p(z))
$$

**高斯假设**：假设 $p(z) = \mathcal{N}(0, I)$，$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$，则 KL 项有闭式解：

$$
KL(q_\phi(z|x) \parallel p(z)) = \frac{1}{2} \sum_{j=1}^{J} \left( \mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1 \right)
$$

其中 $J$ 是潜变量 $z$ 的维度。

## 直观理解

- **VAE 的编码器不是在编码，而是在"描述可能性的分布"**：普通自编码器对每个输入给出一个确定性的编码，VAE 的编码器说"这个输入可能是这些编码值，每个值的可能性如下"。
- **KL 散度像一根橡皮筋**：它把编码分布拉向标准正态分布——拉得太紧会损失重建质量，拉得太松则潜空间结构不好。
- **为什么不直接用自编码器生成**：普通自编码器的潜空间可能有"空洞"——某些区域对应的解码结果是垃圾。VAE 通过迫使潜分布接近标准正态，填满了这些空洞，使得任意采样都有效。
- **VAE 的模糊性问题**：VAE 生成的图像常常比 GAN 模糊，因为 VAE 的 L2 重建损失会鼓励解码器取"所有可能输出的均值"——均值自然就是模糊的。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """变分自编码器（用于图像数据）"""
    def __init__(self, input_dim=784, latent_dim=20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 编码器：输入 -> 潜分布的均值和 log 方差
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # 解码器：潜变量 -> 重建
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),  # 假设像素值在 [0,1]
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        """重参数化技巧：z = mu + sigma * epsilon"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE 损失 = 重建损失 + KL 散度"""
    # 重建损失：二值交叉熵（像素值在 [0,1]）
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL 散度：高斯分布的闭式解
    # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KL, BCE.item(), KL.item()

# 初始化
vae = VAE(input_dim=784, latent_dim=20)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

# 训练步骤
def train_step(vae, optimizer, x):
    vae.train()
    recon_x, mu, logvar = vae(x)
    loss, bce, kl = vae_loss(recon_x, x, mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item(), bce, kl

# 生成新样本
def generate(vae, n_samples=16):
    vae.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, vae.latent_dim)
        samples = vae.decode(z)
    return samples

# 验证
x_dummy = torch.randn(64, 784)
loss_val, bce_val, kl_val = train_step(vae, optimizer, x_dummy)
print(f"训练损失: loss={loss_val:.2f}, BCE={bce_val:.2f}, KL={kl_val:.2f}")
print(f"潜变量维度: {vae.latent_dim}")

samples = generate(vae, 4)
print(f"生成样本形状: {samples.shape}")
```

## 深度学习关联

- **$\beta$-VAE**：在 VAE 损失中引入权重 $\beta$ 来调节 KL 项，$\beta > 1$ 时迫使潜空间更加解耦（Disentangled），学习到独立的语义因子（如旋转、缩放、颜色分离）。
- **VQ-VAE**：将离散化引入 VAE 的潜空间，结合了向量量化和自回归先验（PixelCNN），解决了 VAE 的"后验坍塌"问题，生成的图像比标准 VAE 清晰得多。
- **Stable Diffusion 的 VAE 组件**：现代文本到图像模型使用 VAE 的编码器将像素压缩到潜空间，在潜空间中执行扩散过程，再通过 VAE 解码器恢复为图像——这利用了 VAE 的高效压缩能力。
- **CVAE（条件 VAE）**：通过在编码器和解码器中都加入条件信息 $c$（如类别标签），实现了条件生成，是可控生成的基础框架之一。
