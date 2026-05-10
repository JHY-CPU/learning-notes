# Stable Diffusion与应用

*Latent Diffusion、ControlNet与文生图技术全景*


### 1.1 核心思想


直接在像素空间做扩散计算量巨大（256×256×3 = 196K维度）。**Latent Diffusion**的核心创新是：先用预训练的自编码器将图像压缩到低维潜空间，然后在潜空间中做扩散过程。

图像 x
(512×512×3)
→
编码器 E
(预训练)
→
潜空间 z
(64×64×4)
→
扩散过程
去噪生成
→
解码器 D
重建图像

压缩比：512×512×3 → 64×64×4 = 48倍压缩，大幅减少计算量。


### 1.2 感知压缩自编码器


- 使用
   **VQ-VAE**
   或
   **KL-正则化自编码器**
- 在感知损失（LPIPS）和对抗训练下训练
- 保证重建质量的同时实现高效压缩
- 编码器和解码器独立训练，冻结后供扩散模型使用


### 2.1 三大核心组件


| 组件 | 模型 | 功能 |
| --- | --- | --- |
| 文本编码器 | CLIP Text Encoder (ViT-L/14) | 将文本提示编码为语义向量 |
| 去噪网络 | U-Net (带Cross-Attention) | 在潜空间中逐步去噪 |
| 解码器 | VAE Decoder | 将去噪后的潜变量重建为图像 |


### 2.2 CLIP文本编码器


CLIP（Contrastive Language-Image Pre-training）在4亿图文对上预训练，学习了强大的文本-图像对齐表示：


- 输入：文本提示（如"a photo of a cat on mars"）
- 输出：77×768的序列嵌入（77个token，每个768维）
- SD 1.x使用CLIP ViT-L/14，SD 2.x使用OpenCLIP ViT-H/14
- SDXL同时使用CLIP ViT-L和OpenCLIP ViT-bigG双编码器


### 2.3 U-Net去噪网络


Stable Diffusion的U-Net包含三个关键模块：


- **ResNet块**
   ：提取空间特征（GroupNorm + SiLU + Conv2d）
- **Self-Attention**
   ：建模潜空间内的全局依赖
- **Cross-Attention**
   ：将文本语义注入到去噪过程中（核心！）


```
# Cross-Attention机制
# 查询Q来自图像特征，键K和值V来自文本嵌入
Q = Linear_Q(image_features)   # [B, H*W, dim]
K = Linear_K(text_embeds)      # [B, 77, dim]
V = Linear_V(text_embeds)      # [B, 77, dim]

Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

# 时间步嵌入：将timestep t编码后通过FiLM条件化到每个ResNet块
time_emb = sinusoidal_embedding(t) → MLP → scale, shift
output = GroupNorm(x) * (1 + scale) + shift
```


### 2.4 完整推理流程


```
# Stable Diffusion txt2img完整流程
1. 文本编码: text → CLIP → text_embeddings [1, 77, 768]
2. 初始化: z_T = torch.randn([1, 4, 64, 64])  # 随机潜噪声
3. 去噪循环 (DDIM, 50步):
   for t in reversed(timesteps):
       # U-Net预测噪声
       noise_pred = unet(z_t, t, text_embeddings)
       # CFG引导
       noise_pred_uncond = unet(z_t, t, uncond_embeddings)
       noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)
       # DDIM去噪一步
       z_{t-1} = ddim_step(z_t, noise_pred, t)
4. VAE解码: z_0 → vae_decoder → image [1, 3, 512, 512]
```


### 3.1 核心思想


ControlNet（2023年，张吕敏）通过**零卷积（Zero Convolution）**连接额外的条件控制分支，在不破坏预训练模型的前提下实现精确的结构控制。


### 3.2 架构


- **可训练副本**
   ：克隆U-Net的编码器和中间块作为ControlNet分支
- **零卷积**
   ：权重初始化为0的1×1卷积，初始不影响预训练模型
- **条件输入**
   ：边缘图、深度图、姿态骨架、语义分割等


```
# ControlNet结构
输入条件图 → ControlNet Encoder → Zero Conv → + U-Net各层特征

# 零卷积（Zero Convolution）
nn.Conv2d(in_ch, out_ch, 1)  # 权重和偏置初始化为0
# 训练初期输出为0，不破坏预训练模型
# 训练过程中逐渐学习条件控制信号
```


### 3.3 支持的控制条件


| 条件类型 | 提取方式 | 应用场景 |
| --- | --- | --- |
| Canny边缘 | Canny边缘检测 | 线稿上色、轮廓控制 |
| 深度图 | MiDaS深度估计 | 3D结构控制 |
| 姿态骨架 | OpenPose人体检测 | 人体姿态控制 |
| 语义分割 | 语义分割网络 | 布局控制 |
| 法线图 | 表面法线估计 | 光照和几何控制 |
| 涂鸦 | 用户手绘 | 自由控制 |
| 参考图像 | CLIP图像特征 | 风格迁移 |


### 4.1 img2img


不是从纯噪声开始，而是从一张已有图像的部分加噪版本开始去噪：


```
# img2img流程
1. 编码: image → VAE Encoder → z_0
2. 加噪: z_0 + noise → z_T' (T' < T, 不完全加噪)
3. 去噪: 从z_T'开始去噪到z_0'
4. 解码: z_0' → VAE Decoder → new_image

# denoising_strength: 控制加噪程度
# 0.0 = 原图不变, 1.0 = 等同txt2img
# 0.3-0.7通常效果最佳
```


### 4.2 Inpainting（图像修复）


- 指定遮罩区域，在该区域内重新生成内容
- SD Inpainting模型额外接收遮罩和遮罩区域图像作为输入
- LaMA等专用修复模型可处理大区域缺失


| 版本 | 发布时间 | 文本编码器 | 基础分辨率 | 特点 |
| --- | --- | --- | --- | --- |
| SD 1.5 | 2022.10 | CLIP ViT-L/14 | 512×512 | 成熟生态，LoRA丰富 |
| SD 2.0/2.1 | 2022.11 | OpenCLIP ViT-H/14 | 768×768 | 更大文本编码器 |
| SDXL | 2023.07 | CLIP+OpenCLIP双编码器 | 1024×1024 | Refiner两阶段，质量飞跃 |
| SDXL Turbo | 2023.11 | 同SDXL | 1024×1024 | Adversarial Diffusion Distillation, 1-4步生成 |
| SD 3 | 2024.02 | CLIP+T5-XXL | 1024×1024 | MM-DiT架构，文本理解更强 |
| FLUX.1 | 2024.08 | CLIP+T5 | 多种 | 黑森林团队，开源最强 |


### 6.1 低秩适配（Low-Rank Adaptation）


LoRA冻结预训练模型权重，在每层旁路注入低秩矩阵进行微调：


$$
W' = W + ΔW = W + B·A　　（A∈R^{r×d}, B∈R^{d×r}, r << d）
$$


- 通常只对Attention的Q/K/V投影层添加LoRA
- 秩r通常取4-64，参数量仅为原始模型的0.1%-1%
- 训练速度快，显存需求低（8GB显存即可训练）
- 可叠加多个LoRA实现风格+角色+概念的组合


```
# LoRA微调示例（使用diffusers）
from diffusers import StableDiffusionPipeline
from peft import LoraConfig, get_peft_model

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["to_q", "to_k", "to_v"])
model = get_peft_model(pipe.unet, lora_config)
```


### 7.1 文生图应用


- **Midjourney**
   ：闭源高质量文生图服务
- **DALL-E 3**
   ：OpenAI的文生图模型，集成在ChatGPT中
- **Stable Diffusion WebUI (A1111/ComfyUI)**
   ：开源社区的主流界面


### 7.2 扩展应用


| 应用 | 技术 | 说明 |
| --- | --- | --- |
| 视频生成 | SDV, AnimateDiff | 将SD扩展到时间维度 |
| 3D生成 | DreamFusion, MVDream | 用SD做Score Distillation |
| 图像编辑 | InstructPix2Pix | 通过文本指令编辑图像 |
| 超分辨率 | SD Upscaler | 用SD做高质量超分 |
| 风格迁移 | IP-Adapter | 图像提示替代文本提示 |


<!-- Converted from: 02_Stable Diffusion与应用.html -->
