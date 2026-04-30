import os

titles = [
    "01_生成模型与判别模型的对比",
    "02_GAN 基础：生成器与判别器的博弈论视角",
    "03_GAN 训练不稳定性与模式崩溃 (Mode Collapse)",
    "04_DCGAN：深度卷积生成对抗网络",
    "05_Wasserstein GAN (WGAN) 与 Earth Mover's Distance",
    "06_WGAN-GP：梯度惩罚项的实现细节",
    "07_StyleGAN：风格混合与自适应实例归一化",
    "08_CycleGAN：无配对数据的图像翻译",
    "09_Pix2Pix：基于条件的图像翻译",
    "10_VAE 变分自编码器原理",
    "11_重参数化技巧 (Reparameterization Trick) 推导",
    "12_VAE 损失函数：重构误差与 KL 散度权衡",
    "13_CVAE：条件变分自编码器",
    "14_VQ-VAE：向量量化与离散潜在空间",
    "15_扩散模型 (Diffusion Models) 物理直觉",
    "16_DDPM：去噪扩散概率模型数学推导",
    "17_前向加噪过程与马尔可夫链",
    "18_反向去噪过程与神经网络预测",
    "19_Score Matching 与 Langevin Dynamics",
    "20_DDIM：确定性采样与加速推理",
    "21_Stable Diffusion：Latent Space 压缩技术",
    "22_ControlNet：添加空间约束条件",
    "23_LoRA 在生成模型微调中的应用",
    "24_DreamBooth：个性化主体生成",
    "25_Text-to-Image 模型中的 CLIP 文本编码器",
    "26_超分辨率扩散模型 (Super-Resolution)",
    "27_视频生成：AnimateDiff 原理",
    "28_3D 生成：Point-E 与 Shap-E",
    "29_音频生成：AudioLDM 基础",
    "30_生成模型评估指标：FID 与 IS"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
