# 13_CVAE：条件变分自编码器

## 核心概念
- **条件变分自编码器 (CVAE)**：VAE 的条件版本，在编码器和解码器中都加入条件信息 $c$（如类别标签、文本描述），实现受控生成。
- **条件编码器**：$q_\phi(z|x,c)$ —— 编码器不仅接收输入 $x$，还接收条件 $c$，输出在条件 $c$ 下的后验分布。
- **条件解码器**：$p_\theta(x|z,c)$ —— 解码器根据潜变量 $z$ 和条件 $c$ 生成输出，条件 $c$ 指导生成过程产生符合要求的结果。
- **可控生成**：给定相同的潜变量 $z$ 但不同的条件 $c$，解码器可以产生不同类别的样本，实现了生成内容的可控性。
- **输入输出对齐**：CVAE 天生适合需要条件生成的任务，如文本到图像生成、语音合成（给定文本生成语音）、手写数字生成（给定数字标签）。
- **条件先验**：可以进一步建模条件先验 $p(z|c)$ 而不是标准的 $\mathcal{N}(0,I)$，如果条件 $c$ 与 $z$ 有相关性。

## 数学推导

**CVAE 的 ELBO**：

$$
\log p_\theta(x|c) \geq \mathcal{L}(x, c; \theta, \phi) = \mathbb{E}_{z \sim q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - KL(q_\phi(z|x,c) \parallel p(z|c))
$$

其中 $p(z|c)$ 是条件先验（通常仍取标准正态 $\mathcal{N}(0,I)$，但不必须）。

**与标准 VAE 的对比**：

标准 VAE：$\mathcal{L}(x) = \mathbb{E}_{z \sim q_\phi(z|x)}[\log p_\theta(x|z)] - KL(q_\phi(z|x) \parallel p(z))$

CVAE：$\mathcal{L}(x, c) = \mathbb{E}_{z \sim q_\phi(z|x,c)}[\log p_\theta(x|z,c)] - KL(q_\phi(z|x,c) \parallel p(z|c))$

区别在于：标准 VAE 的生成是完全无条件的（从标准正态采样即可生成），而 CVAE 的生成需要先指定条件 $c$。

**条件信息的注入方式**：
1. **拼接 (Concatenation)**：将条件 $c$ 与 $x$ 或 $z$ 沿着特征维度拼接，最简单直接。
2. **特征调制 (Feature Modulation)**：通过学习仿射变换（类似 AdaIN）将条件信息注入网络各层。
3. **交叉注意力 (Cross-Attention)**：条件 $c$ 作为 Query/Key，特征图作为 Value，进行注意力计算（现代扩散模型的常用方式）。

**CVAE 的条件先验**：如果条件 $c$ 与潜变量 $z$ 存在关联（如不同类别的手写数字对应不同的潜分布），可以建模 $p(z|c) = \mathcal{N}(\mu_\theta(c), \sigma_\theta^2(c))$，其中 $\mu_\theta(c)$ 和 $\sigma_\theta(c)$ 是从条件 $c$ 学习得到的。

## 直观理解
- **CVAE = 有了"主题"的画家**：标准 VAE 像画家自由创作，你不知道他会画什么。CVAE 相当于你告诉画家"画一只猫"（条件），画家根据这个指令创作。
- **条件信息相当于"锚"**：在潜空间中，条件 $c$ 将潜变量 $z$ 引导到某个特定区域。同样一张脸（$z$ 相同），加上"微笑"的条件就笑，加上"悲伤"的条件就哭。
- **为什么需要条件编码器**：$q(z|x,c)$ 意味着编码器在看输入 $x$ 的同时也看到了条件 $c$——它只需要编码"除去条件后剩下的信息"。比如，给定了"数字是 7"的条件，编码器只需要编码笔画的粗细、倾斜度等风格信息。
- **条件先验的意义**：如果条件先验 $p(z|c)$ 不从属于条件 $c$，你的模型可能生成一个"8"但训练标签给的是"3"——条件指导了重建但没能有效约束潜分布。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class CVAE(nn.Module):
    """条件变分自编码器：支持类别条件"""
    def __init__(self, input_dim=784, latent_dim=20, num_classes=10, condition_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # 条件嵌入
        self.condition_embed = nn.Linear(num_classes, condition_dim)
        
        # 编码器：输入 + 条件 -> 潜分布参数
        enc_input_dim = input_dim + condition_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # 解码器：潜变量 + 条件 -> 重建
        dec_input_dim = latent_dim + condition_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )
    
    def encode(self, x, c):
        """条件编码：传入输入和条件"""
        # c: one-hot 编码的类别标签
        c_emb = self.condition_embed(c)
        xc = torch.cat([x, c_emb], dim=1)
        h = self.encoder(xc)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        """条件解码：传入潜变量和条件"""
        c_emb = self.condition_embed(c)
        zc = torch.cat([z, c_emb], dim=1)
        return self.decoder(zc)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar

class CVAELoss(nn.Module):
    """CVAE 损失函数"""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
    
    def forward(self, recon_x, x, mu, logvar):
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss

# 条件生成示例：生成指定数字
def generate_digit(cvae, digit, n_samples=16, latent_dim=20):
    """生成指定数字的图像"""
    cvae.eval()
    with torch.no_grad():
        # 随机采样潜变量
        z = torch.randn(n_samples, latent_dim)
        # 构造 one-hot 条件
        c = torch.zeros(n_samples, 10)
        c[:, digit] = 1.0
        # 条件解码
        samples = cvae.decode(z, c)
    return samples

# 条件潜变量插值：固定条件，插值潜变量
def conditional_interpolation(cvae, digit, z1, z2, steps=8):
    """在给定条件下的潜空间插值"""
    alphas = torch.linspace(0, 1, steps)
    c = torch.zeros(1, 10)
    c[:, digit] = 1.0
    c = c.expand(steps, -1)
    
    imgs = []
    for alpha in alphas:
        z = (1 - alpha) * z1 + alpha * z2
        with torch.no_grad():
            img = cvae.decode(z, c)
        imgs.append(img)
    return torch.stack(imgs)

# 初始化
cvae = CVAE(input_dim=784, latent_dim=20, num_classes=10)
optimizer = torch.optim.Adam(cvae.parameters(), lr=1e-3)
loss_fn = CVAELoss(beta=1.0)

# 模拟训练
x_dummy = torch.randn(32, 784)
c_dummy = torch.eye(10)[torch.randint(0, 10, (32,))]
recon, mu, logvar = cvae(x_dummy, c_dummy)
loss = loss_fn(recon, x_dummy, mu, logvar)
print(f"CVAE 训练损失: {loss.item():.2f}")

# 条件生成测试
samples_3 = generate_digit(cvae, digit=3, n_samples=4)
samples_7 = generate_digit(cvae, digit=7, n_samples=4)
print(f"生成数字 3: {samples_3.shape}")
print(f"生成数字 7: {samples_7.shape}")
print("CVAE 实现了可控的条件生成！")
```

## 深度学习关联
- **文本到图像生成的基础**：CVAE 的结构启发了 StackGAN、AttnGAN 等文本到图像生成模型——将文本编码作为条件，通过多阶段精化生成高分辨率图像。
- **扩散模型中的条件控制**：现代扩散模型（如 Stable Diffusion）本质上可以看作一个"条件去噪自编码器"，通过交叉注意力层将文本嵌入、CLIP 编码等条件注入到 U-Net 中——这是 CVAE 思想在扩散框架中的体现。
- **CVAE + 序列模型**：在语音合成（如 Tacotron）、手写生成、运动序列生成中，CVAE 的条件框架被广泛用于控制生成内容的风格、内容和韵律。
- **Disentanglement 在条件 VAE 中的扩展**：通过巧妙设计条件 $c$ 的不同维度，可以实现对生成样本不同属性的独立控制——如人脸生成中控制年龄、性别、发型等。
