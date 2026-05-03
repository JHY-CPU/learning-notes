# 24_DreamBooth：个性化主体生成

## 核心概念

- **DreamBooth**：一种个性化文本到图像生成方法，用少量（3-5 张）特定主体（如宠物、人物、物品）的图像微调整个扩散模型，使模型学会在文本提示中引用该主体。
- **稀有词标识符 (Unique Identifier)**：用一个稀有 token 序列（如 "sks"）作为主体的代理标识，将该 token 与主体图像绑定。在推理时通过 "a [sks] dog" 生成特定狗狗的图像。
- **先验保留损失 (Prior Preservation Loss)**：防止灾难性遗忘的关键技术——在微调时同时使用模型自身生成的同类图像，确保模型不会忘记"狗的一般样子"而只记得"我的狗的样子"。
- **模型微调范围**：DreamBooth 微调整个扩散模型的全部参数（U-Net + 文本编码器），计算量大但效果好（相比 LoRA 等轻量方法）。
- **主体编辑能力**：训练后不仅能用文本提示生成该主体的新图像（"我的猫在月球上"），还能保持主体的身份特征、纹理细节。
- **过拟合风险**：只有 3-5 张训练图像，极易过拟合。先验保留损失和数据增强是缓解过拟合的关键手段。

## 数学推导

**DreamBooth 训练损失**：

标准扩散模型损失 + 先验保留损失：

$$
\mathbb{E}_{x, c, \epsilon, t} \left[ w_t \| \hat{x}_\theta(\alpha_t x + \sigma_t \epsilon, c) - x \|_2^2 \right] + \lambda \mathbb{E}_{x_{\text{pr}}, c_{\text{pr}}, \epsilon', t'} \left[ w_{t'} \| \hat{x}_\theta(\alpha_{t'} x_{\text{pr}} + \sigma_{t'} \epsilon', c_{\text{pr}}) - x_{\text{pr}} \|_2^2 \right]
$$

第一项：用特定主体的图像 $x$ 和文本标识符 $c$="a [sks] dog" 更新模型

第二项（先验保留）：用模型自己生成的同类图像 $x_{\text{pr}}$ 和类别文本 $c_{\text{pr}}$="a dog" 更新模型

$\lambda$ 控制先验保留的强度（通常 $\lambda = 1$）。

**先验保留损失的生成过程**：

冻结模型权重 $\theta_{\text{old}}$，生成先验图像：

$$
x_{\text{pr}} \sim p_{\theta_{\text{old}}}(x | c_{\text{pr}})
$$

然后用这些图像来约束 $\theta$ 的更新，确保：

$$
p_\theta(x | c_{\text{pr}}) \approx p_{\theta_{\text{old}}}(x | c_{\text{pr}})
$$

同时保持：

$$
p_\theta(x | c) \approx p_{\text{data}}(x | c)
$$

其中 $p_{\text{data}}(x | c)$ 是特定主体图像的经验分布。

## 直观理解

- **DreamBooth = 给 AI 介绍你的宠物**：你给 AI 展示 5 张你狗狗的照片（训练），告诉它"这是 sks 狗"。以后你说"画一只 sks 狗在公园里玩"，AI 就知道画你的狗，不是随便一只狗。
- **先验保留损失 = 不要只认识我的狗**：如果只用 5 张自家狗的照片训练，AI 会忘记世上还有其他狗——你说"画一只狗"，它会画成你家狗的样子。先验保留损失通过让 AI 同时复习"一般狗长什么样"来防止这种遗忘。
- **稀有 token 的作用**：使用稀有 token "sks" 而非"my_dog" 的目的是——"my_dog" 这个词已经被模型用来表示各种概念，难以绑定到特定个体。"sks" 像一个空的挂钩，可以干净地挂上新的概念。
- **为什么需要多张照片**：3-5 张不同角度、不同环境下的照片让模型理解"同一个主体在不同条件下的不变特征"——就像人类通过多角度观察认识一个人。

## 代码示例

```python
import torch
import torch.nn as nn

class DreamBoothTrainer:
    """
    DreamBooth 训练器简化实现
    
    实际实现需要完整的 Stable Diffusion 模型，
    这里展示核心逻辑。
    """
    def __init__(self, unet, text_encoder, tokenizer, noise_scheduler):
        self.unet = unet
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.noise_scheduler = noise_scheduler
        
        # 优化所有参数
        self.optimizer = torch.optim.Adam(
            list(unet.parameters()) + list(text_encoder.parameters()),
            lr=5e-6
        )
    
    def encode_text(self, text):
        """将文本转换为嵌入"""
        tokens = self.tokenizer(text, return_tensors="pt", padding=True)
        return self.text_encoder(tokens.input_ids)
    
    def compute_prior_preservation_loss(self, class_prompt="a dog", num_images=4):
        """
        先验保留损失：用冻结模型生成同类图像，计算重建损失
        
        关键步骤：
        1. 冻结当前模型权重
        2. 用类别提示生成一批图像（如"a dog"）
        3. 解冻模型，在这些图像上计算扩散损失
        """
        # 用冻结的模型生成先验图像
        with torch.no_grad():
            # 实际需要调用完整的采样过程
            # 此处用随机数据模拟
            prior_images = torch.randn(num_images, 3, 512, 512)
        
        # 编码类别提示
        prior_emb = self.encode_text(class_prompt)
        
        # 在先验图像上计算扩散损失
        prior_noise = torch.randn_like(prior_images)
        t = torch.randint(0, 1000, (num_images,))
        noisy_prior = self.noise_scheduler.add_noise(prior_images, prior_noise, t)
        
        noise_pred = self.unet(noisy_prior, t, prior_emb).sample
        prior_loss = nn.functional.mse_loss(noise_pred, prior_noise)
        
        return prior_loss
    
    def train_step(self, subject_images, subject_prompt="a sks dog", 
                   class_prompt="a dog", prior_loss_weight=1.0):
        """
        DreamBooth 训练步骤
        
        参数:
            subject_images: 特定主体的图像 [B, C, H, W]
            subject_prompt: 包含稀有标识符的提示
            class_prompt: 类别提示用于先验保留
            prior_loss_weight: 先验保留损失权重
        """
        batch_size = subject_images.size(0)
        
        # 1. 主体损失
        subject_emb = self.encode_text(subject_prompt)
        
        noise = torch.randn_like(subject_images)
        t = torch.randint(0, 1000, (batch_size,))
        noisy_subject = self.noise_scheduler.add_noise(subject_images, noise, t)
        
        noise_pred = self.unet(noisy_subject, t, subject_emb).sample
        subject_loss = nn.functional.mse_loss(noise_pred, noise)
        
        # 2. 先验保留损失
        prior_loss = self.compute_prior_preservation_loss(
            class_prompt, batch_size
        )
        
        # 总损失
        total_loss = subject_loss + prior_loss_weight * prior_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'subject_loss': subject_loss.item(),
            'prior_loss': prior_loss.item(),
        }
    
    def save_pretrained(self, path):
        """保存微调后的模型"""
        # 实际需要保存 unet 和 text_encoder 的状态字典
        print(f"模型已保存到 {path}")

# 使用示例
print("=== DreamBooth 训练流程 ===")
print()
print("训练数据: 3-5 张特定主体的图像")
print("训练提示: 'a sks dog' (sks 是稀有标识符)")
print("先验提示: 'a dog' (类别提示)")
print()
print("DreamBooth 损失组成:")
print("  1. 主体重建损失: 学习特定主体的外观")
print("  2. 先验保留损失: 保持对同类物体的泛化能力")
print("  3. 权重 lambda: 平衡两项损失（通常 lambda=1）")
print()

# 参数分析
subject_images = 5
prior_images = 200  # 实际中生成更多先验图像
total_images_per_step = subject_images + prior_images
print(f"每步训练图像: 主体 {subject_images} 张 + 先验 {prior_images} 张 = {total_images_per_step} 张")
print(f"主体图像占比: {subject_images/total_images_per_step*100:.1f}%")
```

## 深度学习关联

- **DreamBooth 与 LoRA 的对比**：DreamBooth 全量微调（效果好但文件大~2GB），LoRA 高效微调（效果略逊但文件小~10MB）。实践中常组合使用——用 DreamBooth 训练主体，用 LoRA 形式导出权重。
- **Textual Inversion**：DreamBooth 的轻量替代——不修改模型权重，只在文本嵌入空间中学习一个新 token 的嵌入向量（几 KB 大小）。效果不如 DreamBooth 但极其轻量。
- **Custom Diffusion**：介于 DreamBooth 和 Textual Inversion 之间——只微调交叉注意力层中的 K 和 V 投影矩阵，参数量远小于 DreamBooth 但效果接近。
- **主题驱动的视频生成**：DreamBooth 的技术被扩展到视频领域（如 Dreamix, VideoDreamBooth），只需少量主体图像就能在视频生成中保持主体的身份一致性。
