# 06_Generative_AI

生成式AI：GAN、VAE、扩散模型、ControlNet、评估指标

共 50 篇笔记

| 编号 | 笔记 |
|------|------|
| 01 | [生成模型与判别模型的对比](01_生成模型与判别模型的对比.md) |
| 02 | [GAN 基础：生成器与判别器的博弈论视角](02_GAN 基础：生成器与判别器的博弈论视角.md) |
| 03 | [GAN 训练不稳定性与模式崩溃 (Mode Collapse)](03_GAN 训练不稳定性与模式崩溃 (Mode Collapse).md) |
| 04 | [DCGAN：深度卷积生成对抗网络](04_DCGAN：深度卷积生成对抗网络.md) |
| 05 | [Wasserstein GAN (WGAN) 与 Earth Mover's Distance](05_Wasserstein GAN (WGAN) 与 Earth Mover's Distance.md) |
| 06 | [WGAN-GP：梯度惩罚项的实现细节](06_WGAN-GP：梯度惩罚项的实现细节.md) |
| 07 | [StyleGAN：风格混合与自适应实例归一化](07_StyleGAN：风格混合与自适应实例归一化.md) |
| 08 | [CycleGAN：无配对数据的图像翻译](08_CycleGAN：无配对数据的图像翻译.md) |
| 09 | [Pix2Pix：基于条件的图像翻译](09_Pix2Pix：基于条件的图像翻译.md) |
| 10 | [VAE 变分自编码器原理](10_VAE 变分自编码器原理.md) |
| 11 | [重参数化技巧 (Reparameterization Trick) 推导](11_重参数化技巧 (Reparameterization Trick) 推导.md) |
| 12 | [VAE 损失函数：重构误差与 KL 散度权衡](12_VAE 损失函数：重构误差与 KL 散度权衡.md) |
| 13 | [CVAE：条件变分自编码器](13_CVAE：条件变分自编码器.md) |
| 14 | [VQ-VAE：向量量化与离散潜在空间](14_VQ-VAE：向量量化与离散潜在空间.md) |
| 15 | [扩散模型 (Diffusion Models) 物理直觉](15_扩散模型 (Diffusion Models) 物理直觉.md) |
| 16 | [DDPM：去噪扩散概率模型数学推导](16_DDPM：去噪扩散概率模型数学推导.md) |
| 17 | [前向加噪过程与马尔可夫链](17_前向加噪过程与马尔可夫链.md) |
| 18 | [反向去噪过程与神经网络预测](18_反向去噪过程与神经网络预测.md) |
| 19 | [Score Matching 与 Langevin Dynamics](19_Score Matching 与 Langevin Dynamics.md) |
| 20 | [DDIM：确定性采样与加速推理](20_DDIM：确定性采样与加速推理.md) |
| 21 | [Stable Diffusion：Latent Space 压缩技术](21_Stable Diffusion：Latent Space 压缩技术.md) |
| 22 | [ControlNet：添加空间约束条件](22_ControlNet：添加空间约束条件.md) |
| 23 | [LoRA 在生成模型微调中的应用](23_LoRA 在生成模型微调中的应用.md) |
| 24 | [DreamBooth：个性化主体生成](24_DreamBooth：个性化主体生成.md) |
| 25 | [Text-to-Image 模型中的 CLIP 文本编码器](25_Text-to-Image 模型中的 CLIP 文本编码器.md) |
| 26 | [超分辨率扩散模型 (Super-Resolution)](26_超分辨率扩散模型 (Super-Resolution).md) |
| 27 | [视频生成：AnimateDiff 原理](27_视频生成：AnimateDiff 原理.md) |
| 28 | [3D 生成：Point-E 与 Shap-E](28_3D 生成：Point-E 与 Shap-E.md) |
| 29 | [音频生成：AudioLDM 基础](29_音频生成：AudioLDM 基础.md) |
| 30 | [生成模型评估指标：FID 与 IS](30_生成模型评估指标：FID 与 IS.md) |
| 31 | [Inception Score (IS) 的计算逻辑](31_Inception Score (IS) 的计算逻辑.md) |
| 32 | [Precision and Recall for GANs](32_Precision and Recall for GANs.md) |
| 33 | [流形假设与生成模型的关系](33_流形假设与生成模型的关系.md) |
| 34 | [能量基模型 (EBM) 基础](34_能量基模型 (EBM) 基础.md) |
| 35 | [扩散模型的 SDE 视角解释](35_扩散模型的 SDE 视角解释.md) |
| 36 | [Consistency Models：一步生成技术](36_Consistency Models：一步生成技术.md) |
| 37 | [Latent Consistency Models (LCM)](37_Latent Consistency Models (LCM).md) |
| 38 | [Guidance Scale：分类器引导与无分类器引导](38_Guidance Scale：分类器引导与无分类器引导.md) |
| 39 | [Cross-Attention 在文本控制中的作用](39_Cross-Attention 在文本控制中的作用.md) |
| 40 | [生成式 AI 的版权与伦理问题](40_生成式 AI 的版权与伦理问题.md) |
| 41 | [Deepfake 检测技术与防御](41_Deepfake 检测技术与防御.md) |
| 42 | [提示词工程 (Prompt Engineering) 技巧](42_提示词工程 (Prompt Engineering) 技巧.md) |
| 43 | [Negative Prompt 的实现机制](43_Negative Prompt 的实现机制.md) |
| 44 | [Inpainting 与 Outpainting 算法原理](44_Inpainting 与 Outpainting 算法原理.md) |
| 45 | [Model Merging 模型合并技术](45_Model Merging 模型合并技术.md) |
| 46 | [Distilled Diffusion 蒸馏加速](46_Distilled Diffusion 蒸馏加速.md) |
| 47 | [Autoregressive vs Diffusion 生成范式对比](47_Autoregressive vs Diffusion 生成范式对比.md) |
| 48 | [Masked Generative Modeling (MAGVIT)](48_Masked Generative Modeling (MAGVIT).md) |
| 49 | [多模态大模型 (LMM) 的生成能力](49_多模态大模型 (LMM) 的生成能力.md) |
| 50 | [世界模型 (World Models) 与生成式 AI](50_世界模型 (World Models) 与生成式 AI.md) |
