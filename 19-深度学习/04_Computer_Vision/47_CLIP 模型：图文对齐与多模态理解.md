# 47_CLIP 模型：图文对齐与多模态理解

## 核心概念

- **CLIP（Contrastive Language-Image Pre-training）**：OpenAI (2021) 提出的多模态模型，使用4亿图文对进行对比学习，将图像和文本映射到同一语义空间。CLIP展现出强大的零样本图像分类和跨模态理解能力。
- **对比学习目标**：对于一个batch中的 $N$ 个图文对，CLIP学习最大化正确图文对之间的余弦相似度，最小化不正确图文对之间的相似度。使用对称的InfoNCE损失。
- **双塔架构（Two-Tower Architecture）**：CLIP包含两个独立的编码器——图像编码器（ResNet或ViT）和文本编码器（Transformer）。两种模态编码到相同的嵌入维度。
- **零样本分类能力**：CLIP可以通过文本提示（如"a photo of a cat"）为图像分类——将图像编码与所有类别文本编码进行对比，选择最相似的类别，不需要任何训练样本。
- **Prompt Engineering**：零样本分类的效果高度依赖于文本提示的设计。使用模板如"A photo of a {class}"比直接使用类名更好。集成多个模板（如"a bad photo of a {class}"、"a blurry photo of a {class}"）可以进一步提升精度。
- **可迁移性**：CLIP的视觉特征可以用于多种下游任务（目标检测、语义分割、视频理解等），作为通用的视觉-语言表示。

## 数学推导

**CLIP的对比损失（对称InfoNCE）：**

给定一个batch的 $N$ 个图文对 $(I_i, T_i)$，图像和文本编码器分别输出 $f(I_i)$ 和 $g(T_i)$，均为 $D$ 维向量。计算相似度矩阵：
$$
S_{ij} = f(I_i)^T g(T_j) \cdot e^{\tau}
$$

其中 $\tau$ 是可学习的温度参数。

**图像到文本的损失（第 $i$ 个图像与对应文本的匹配概率）：**
$$
\mathcal{L}_{i \to t} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ij})}
$$

**文本到图像的损失：**
$$
\mathcal{L}_{t \to i} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(S_{ii})}{\sum_{j=1}^N \exp(S_{ji})}
$$

**总损失（对称）：**
$$
\mathcal{L} = \frac{\mathcal{L}_{i \to t} + \mathcal{L}_{t \to i}}{2}
$$

**CLIP的零样本分类过程：**
$$
\text{prediction} = \arg\max_{c \in \mathcal{C}} \cos(f(I), g(\text{"a photo of a "} + c))
$$

其中 $\mathcal{C}$ 是类别名称集合，$\cos$ 是余弦相似度。

## 直观理解

CLIP的工作方式可以想象为一个"双语翻译器"——不过它的"语言"是图像和自然语言。CLIP通过同时看4亿张图片和对应的文字描述（"一只坐在沙发上的橘猫"、"夕阳下的金门大桥"等），学会了"图像中有什么"和"语言如何描述它"之间的对应关系。

关键是CLIP学习的是"对齐"而不是"生成"——它不生成文字描述，而是把图像特征和文本特征映射到同一个向量空间中。在这个空间中，"猫"的文字特征和猫的照片的视觉特征非常接近，而和"狗"的文字特征相对较远。

零样本分类时，CLIP像是一个"开卷考试"——给定一张图片和一组可能的标签（"猫"、"狗"、"鸟"），CLIP用视觉编码器读取图片（视觉特征），用文本编码器读取所有标签（文本特征），然后问"哪个文本特征与这张图片的视觉特征最接近？"

## 代码示例

```python
import torch
import torch.nn as nn

class CLIP(nn.Module):
    """简化版 CLIP 双塔模型"""
    def __init__(self, img_dim=512, text_dim=512, embed_dim=256, 
                 vision_width=768, text_width=512):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # 图像编码器 (简化: 用线性层替代ViT)
        self.visual_proj = nn.Sequential(
            nn.Linear(vision_width, embed_dim),
        )
        # 文本编码器 (简化: 用线性层替代Transformer)
        self.text_proj = nn.Sequential(
            nn.Linear(text_width, embed_dim),
        )

    def forward(self, image_features, text_features):
        # image_features: (B, vision_width)
        # text_features: (B, text_width)
        
        # 投影到共享嵌入空间
        img_embed = self.visual_proj(image_features)
        txt_embed = self.text_proj(text_features)
        
        # L2归一化
        img_embed = img_embed / img_embed.norm(dim=-1, keepdim=True)
        txt_embed = txt_embed / txt_embed.norm(dim=-1, keepdim=True)
        
        # 计算相似度矩阵
        logits = img_embed @ txt_embed.T * self.temperature.exp()
        return logits

    def contrastive_loss(self, logits):
        """对称对比损失"""
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size).to(logits.device)
        
        loss_i = nn.functional.cross_entropy(logits, labels)  # image->text
        loss_t = nn.functional.cross_entropy(logits.T, labels)  # text->image
        return (loss_i + loss_t) / 2

# 测试
model = CLIP(vision_width=768, text_width=512, embed_dim=256)

# 模拟一个batch的4个图文对
img_feats = torch.randn(4, 768)
txt_feats = torch.randn(4, 512)

logits = model(img_feats, txt_feats)
loss = model.contrastive_loss(logits)
print(f"相似度矩阵:\n{logits}")
print(f"对比损失: {loss.item():.4f}")

# 零样本分类模拟
class_names = ['cat', 'dog', 'bird', 'fish']
# 文本提示
prompts = [f"a photo of a {name}" for name in class_names]
# 模拟文本特征 (实际需用文本编码器)
text_embeds = torch.randn(4, 256)
# 模拟图像特征
image_embed = torch.randn(1, 256)

# 零样本分类
similarities = image_embed @ text_embeds.T
pred_class = class_names[similarities.argmax().item()]
print(f"零样本分类预测: {pred_class}")

print("\nCLIP的特性与局限:")
print("- 零样本能力强: 无需微调即可用于新类别")
print("- 对分布偏移鲁棒: 在自然分布偏移下表现稳定")
print("- 对细粒度分类有限: 无法区分紧密相似的类别")
print("- 对抽象概念有限: 在counting、spatial关系等任务上表现不佳")
```

## 深度学习关联

- **视觉-语言模型的基础架构**：CLIP的双塔对比学习架构成为视觉-语言基础模型的标准范式。后续的SigLIP（使用sigmoid损失替代softmax）、OpenCLIP（开源复现）、EVA-CLIP（更大规模）等都在此基础上改进。
- **许多多模态系统的核心组件**：CLIP被广泛用作许多多模态系统的基础组件——文本到图像生成（Stable Diffusion使用CLIP的文本编码器）、图像描述（作为视觉编码器）、多模态对话（如GPT-4V基于CLIP类模型）等。
- **提示工程与零样本泛化**：CLIP验证了"通过自然语言提示迁移到新任务"的有效性，推动了视觉-语言提示学习（CoOp、CoCoOp、MaPLe）等研究方向的发展，使预训练模型能通过"学习提示"而非"微调整个模型"适配下游任务。
