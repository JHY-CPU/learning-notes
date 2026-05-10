# VAE变分自编码器

*从自编码器到变分推断 — 概率生成模型的优雅框架*


### 1.1 基本自编码器（AE）


自编码器由编码器和解码器组成，通过压缩-重建过程学习数据的低维表示：

输入 x
→
编码器
h = f(x)
→
潜码 z
(瓶颈层)
→
解码器
x̂ = g(z)
→
重建 x̂

**损失函数**：L = ||x - x̂||²（均方误差）或交叉熵


### 1.2 普通AE的问题


- 潜空间没有明确的结构，可能产生"空洞"
- 无法直接采样生成新数据（不知道采哪些z有意义）
- 容易过拟合，学不到有意义的表示


### 2.1 核心思想


VAE（Variational Autoencoder, 2014）不是将输入映射到一个确定的潜码z，而是映射到一个**概率分布**（通常是高斯分布），然后从该分布中采样z。这使得潜空间连续且完整，可以自由采样生成新数据。

输入 x
→
编码器
输出 μ, σ
→
重参数化
z = μ + σ⊙ε
→
解码器
P(x|z)
→
重建 x̂

### 2.2 变分推断（Variational Inference）


目标：最大化数据的对数似然 log P(x)。但真实后验 P(z|x) 通常不可计算，因此用变分分布 Q(z|x)（由编码器参数化）来近似。


### 2.3 ELBO（Evidence Lower Bound）


通过变分推断，可以推导出对数似然的下界（ELBO）：


$$
log P(x) ≥ E_Q(z|x)[log P(x|z)] - KL(Q(z|x) || P(z)) = ELBO
$$


- **重建项**
   ：E_Q[log P(x|z)] — 解码器重建数据的能力
- **KL散度项**
   ：KL(Q(z|x) || P(z)) — 编码器输出分布与先验的接近程度
- 先验 P(z) 通常设为标准正态分布 N(0, I)


最大化ELBO等价于同时最大化重建质量和最小化后验与先验的KL散度。


### 2.4 重参数化技巧（Reparameterization Trick）


**问题**：从Q(z|x)中采样z的操作不可微分，梯度无法反向传播。


**解决**：将随机性转移到一个独立的噪声变量ε：


$$
z = μ + σ ⊙ ε,　其中 ε ~ N(0, I)
$$


μ和σ是编码器输出的确定性参数，ε是独立的随机噪声。这样z关于μ和σ的梯度就可以正常计算了。


```
# VAE重参数化技巧
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)     # 均值
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # 对数方差

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # σ = exp(0.5·log(σ²))
        eps = torch.randn_like(std)     # ε ~ N(0, I)
        return mu + eps * std            # z = μ + σ⊙ε

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# VAE损失函数
def vae_loss(x, x_recon, mu, logvar):
    # 重建损失（BCE或MSE）
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL散度（闭式解）
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss
```


### 3.1 推导


当Q(z|x)和P(z)都是高斯分布时，KL散度有闭式解：


$$
KL(N(μ, σ²) || N(0, 1)) = -½ Σ (1 + log(σ²) - μ² - σ²)
$$


其中：


- μ²：编码器预测均值的平方（鼓励均值接近0）
- σ²：方差项（鼓励方差接近1）
- log(σ²)：对数方差项（控制方差不过大）


这就是代码中 `kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())` 的来源。


### 4.1 动机


标准VAE学到的潜变量往往是纠缠的（entangled），即一个潜变量同时控制多个语义属性。**β-VAE**通过加大KL项的权重来鼓励解耦（disentangled）表示。


$$
L = E[log P(x|z)] - β · KL(Q(z|x) || P(z))　　（β > 1）
$$


### 4.2 β的作用


- **β = 1**
   ：标准VAE，重建质量好但表示纠缠
- **β > 1**
   ：更强的正则化，迫使每个潜维度独立控制一个生成因素
- **β过大会导致**
   ：重建质量下降（后验塌缩）


### 4.3 相关变体


| 模型 | 改进 |
| --- | --- |
| β-VAE H | 引入容量参数C，逐渐增加KL约束强度 |
| FactorVAE | 用总相关（TC）惩罚替代KL，直接最小化维度间依赖 |
| DIP-VAE | 正则化后验的二阶统计量 |
| β-TCVAE | 分解KL散度，单独惩罚总相关项 |


### 5.1 动机


标准VAE的连续潜变量存在"后验塌缩"问题，且不适合建模离散结构（如语言）。**VQ-VAE（Vector Quantized VAE）**使用**离散的码本（codebook）**替代连续潜变量。


### 5.2 核心机制


```
# VQ-VAE结构
编码器输出: z_e(x) ∈ R^D
码本: e ∈ R^(K×D) （K个离散码向量）

# 向量量化：找到码本中最近的码
z_q = argmin_{e_k} ||z_e(x) - e_k||

# 解码器输入: z_q（码本中的离散向量）

# 梯度处理：直通估计器（Straight-Through Estimator）
# 前向: 输出 z_q
# 反向: 梯度直接从解码器传到编码器（绕过离散量化步骤）
```


### 5.3 损失函数


$$
L = ||x - D(e(z))||² + ||sg(z_e) - e||² + β·||z_e - sg(e)||²
$$


- **重建损失**
   ：||x - D(e(z))||²
- **码本损失**
   ：||sg(z_e) - e||² — 将码向量拉向编码器输出（sg=stop_gradient）
- **承诺损失**
   ：β·||z_e - sg(e)||² — 将编码器输出拉向码向量


### 5.4 应用


- **DALL-E v1**
   ：用VQ-VAE将图像编码为离散token，然后用GPT生成
- **AudioLM / SoundStream**
   ：音频的离散编码
- 与自回归模型（Transformer）结合效果好


### 6.1 基本原理


CVAE在编码器和解码器中都加入条件信息c，实现条件生成：


$$
L = E[log P(x|z,c)] - KL(Q(z|x,c) || P(z|c))
$$


- 编码器：Q(z|x,c)，根据输入x和条件c推断潜变量
- 解码器：P(x|z,c)，根据潜变量z和条件c生成样本
- 应用：类别条件生成、属性控制生成


### 6.2 应用场景


| 应用 | 条件c | 生成目标 |
| --- | --- | --- |
| 手写数字生成 | 数字类别 | 对应数字的图像 |
| 图像编辑 | 属性标签 | 修改指定属性后的图像 |
| 药物设计 | 分子性质 | 具有指定性质的分子结构 |


| 特性 | GAN | VAE |
| --- | --- | --- |
| 理论框架 | 博弈论（极小极大博弈） | 概率推断（最大似然） |
| 生成质量 | 通常更锐利、更逼真 | 通常较模糊 |
| 训练稳定性 | 较难训练，模式崩塌 | 训练稳定 |
| 潜空间结构 | 无明确约束 | 有明确概率结构 |
| 似然计算 | 无法直接计算 | 可估计ELBO |
| 多样性 | 可能模式崩塌 | 覆盖度更好 |
| 推理速度 | 单次前向传播 | 单次前向传播 |
| 编码能力 | 无（标准GAN没有编码器） | 有（编码器可推断潜码） |


> **Note:** **VAE-GAN：**
> 结合两者优势——用VAE的编码器-解码器结构+GAN的判别器提升生成质量。后续的
> **BEGAN**
> 、
> **VAEBM**
> 等也探索了二者的融合。


<!-- Converted from: 02_VAE变分自编码器.html -->
