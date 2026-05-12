# 生成式AI

## 一、变分自编码器（VAE）

### 1.1 原理

VAE通过学习数据的潜在分布来生成新样本。

**编码器**：$q_\phi(z|x)$ 近似后验分布
**解码器**：$p_\theta(x|z)$ 从潜在变量重建数据

**损失函数（ELBO）：**
$$\mathcal{L} = -\mathbb{E}_{q(z|x)}[\log p(x|z)] + D_{KL}(q(z|x) \| p(z))$$

- 第一项：重建损失
- 第二项：KL散度，正则化潜在空间

### 1.2 重参数化技巧

$$z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

使得采样操作可导，支持反向传播。

---

## 二、生成对抗网络（GAN）

### 2.1 基本框架

- **生成器 G**：从噪声生成假样本
- **判别器 D**：区分真假样本

**对抗训练目标：**
$$\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$$

### 2.2 GAN变体

| 模型 | 改进 |
|------|------|
| DCGAN | CNN架构，稳定训练 |
| WGAN | Wasserstein距离，解决模式崩溃 |
| WGAN-GP | 梯度惩罚替代权重裁剪 |
| StyleGAN | 风格分离，高质量人脸生成 |
| CycleGAN | 无配对数据的图像转换 |
| Pix2Pix | 配对数据的图像到图像转换 |

### 2.3 训练技巧

- 标签平滑：真标签用0.9代替1.0
- 噪声注入：在判别器输入中添加噪声
- 谱归一化：稳定判别器训练
- 渐进式训练：从低分辨率逐步增加

---

## 三、扩散模型（Diffusion Models）

### 3.1 前向过程

逐步添加高斯噪声：
$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

### 3.2 反向过程

学习去噪：
$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

### 3.3 代表模型

- **DDPM**：去噪扩散概率模型
- **DDIM**：加速采样
- **Stable Diffusion**：在潜在空间做扩散，大幅降低计算量
- **DALL-E 2/3**：文本到图像生成
- **Midjourney**：商业级图像生成

---

## 四、大语言模型（LLM）

### 4.1 规模定律

模型性能随参数量、数据量、计算量呈幂律提升。

### 4.2 涌现能力

当模型规模超过临界点，出现训练时未显式教授的能力：
- 上下文学习（In-Context Learning）
- 思维链推理（Chain-of-Thought）
- 指令跟随（Instruction Following）

### 4.3 代表模型

GPT-4、Claude、LLaMA、Gemini、Mixtral等。

### 4.4 训练流程

1. **预训练**：大规模无标注语料上自回归训练
2. **SFT（监督微调）**：指令-回答对上微调
3. **RLHF**：通过人类反馈强化学习对齐
4. **DPO**：直接偏好优化，简化RLHF
