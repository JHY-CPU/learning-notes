# CLIP：对比语言-图像预训练

## 1. CLIP 简介

CLIP (Contrastive Language-Image Pre-training) 由 OpenAI 于 2021 年提出，
是视觉-语言预训练领域的里程碑工作。CLIP 通过在 4 亿 (image, text) 对上进行对比学习，
实现了强大的**零样本 (zero-shot)** 图像分类能力。

核心思想：**将图像和文本映射到共享的嵌入空间，使匹配的图文对靠近，不匹配的远离。**

## 2. 双塔架构

CLIP 采用独立的双编码器架构：

```
图像 x_v → Vision Encoder (ViT/ResNet) → 线性投影 → 图像嵌入 v
文本 x_t → Text Encoder (Transformer) → 线性投影 → 文本嵌入 t
```

### 2.1 视觉编码器

支持两种选择：

| 架构 | 模型规模 | 输入分辨率 | ImageNet 零样本 Top-1 |
|------|---------|-----------|---------------------|
| ResNet-50 | 102M | 224 | 59.6% |
| ResNet-101 | 151M | 224 | 62.2% |
| ViT-B/32 | 151M | 224 | 63.2% |
| ViT-B/16 | 150M | 224 | 68.3% |
| ViT-L/14 | 428M | 224 | 75.3% |
| ViT-L/14@336px | 428M | 336 | 76.2% |

### 2.2 文本编码器

标准的 12 层 Transformer 编码器，最大序列长度 77 tokens。

```python
import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class CLIPModel(nn.Module):
    def __init__(self, embed_dim=512, vision_width=768, text_width=512,
                 vocab_size=49408, max_seq_len=77):
        super().__init__()

        # 视觉编码器 (简化 ViT)
        self.visual = vit_b_16(image_size=224)
        self.visual.head = nn.Linear(vision_width, embed_dim)

        # 文本编码器
        self.token_embedding = nn.Embedding(vocab_size, text_width)
        self.position_embedding = nn.Embedding(max_seq_len, text_width)
        self.text_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(text_width, 8, batch_first=True),
            num_layers=12
        )
        self.text_proj = nn.Linear(text_width, embed_dim)

        # 可学习温度参数
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1/0.07)))

        self.max_seq_len = max_seq_len

    def encode_image(self, image):
        features = self.visual(image)
        return F.normalize(features, dim=-1)

    def encode_text(self, tokens):
        seq_len = tokens.size(1)
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0)

        x = self.token_embedding(tokens) + self.position_embedding(positions)
        x = self.text_transformer(x)

        # 取 [EOS] token 的特征 (通常是序列中最后一个非 padding token)
        eos_pos = (tokens != 0).sum(dim=1) - 1
        x = x[torch.arange(x.size(0)), eos_pos]
        x = self.text_proj(x)
        return F.normalize(x, dim=-1)

    def forward(self, images, texts):
        image_features = self.encode_image(images)
        text_features = self.encode_text(texts)

        # 余弦相似度
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.T

        # 对称对比损失
        batch_size = logits.size(0)
        labels = torch.arange(batch_size, device=logits.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2
```

## 3. 对比损失

CLIP 使用**对称的 InfoNCE 损失**：

$$\mathcal{L} = \frac{1}{2} \left( \mathcal{L}_{i \to t} + \mathcal{L}_{t \to i} \right)$$

图像到文本：

$$\mathcal{L}_{i \to t} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_i) \cdot \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{v}_i, \mathbf{t}_j) \cdot \tau)}$$

文本到图像：

$$\mathcal{L}_{t \to i} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_i) \cdot \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{t}_i, \mathbf{v}_j) \cdot \tau)}$$

其中 $\tau = e^{\log \tau}$ 为可学习的温度参数。

## 4. 零样本分类

CLIP 最强大的能力是无需任何训练即可执行图像分类：

### 4.1 流程

```
1. 构建文本提示: "a photo of a {class}"
2. 文本编码器获取所有类别的文本嵌入
3. 图像编码器获取图像嵌入
4. 计算图像与所有类别文本的相似度
5. 选择最相似的类别作为预测
```

```python
@torch.no_grad()
def zero_shot_classify(model, image, class_names, templates=None):
    if templates is None:
        templates = ["a photo of a {}"]

    # 构建所有提示
    texts = [t.format(name) for t in templates for name in class_names]
    text_tokens = clip_tokenizer(texts)

    # 编码
    image_features = model.encode_image(image.unsqueeze(0))
    text_features = model.encode_text(text_tokens)

    # 相似度 -> 预测
    similarity = image_features @ text_features.T
    probs = (similarity * model.logit_scale.exp()).softmax(dim=-1)

    # 聚合 (每个类可能有多个模板)
    num_templates = len(templates)
    probs = probs.reshape(-1, num_templates, len(class_names)).mean(dim=1)

    return probs.argmax(dim=-1)
```

### 4.2 Prompt Engineering

不同的 prompt 模板对性能影响显著：

| Prompt 模板 | ImageNet Top-1 |
|------------|---------------|
| "a photo of a {class}" | 75.3% |
| "a blurry photo of a {class}" | +0.5% |
| "a photo of the large {class}" | +0.3% |
| ensembling 80 templates | 76.2% |

## 5. 训练细节

```python
# 训练超参数 (ViT-L/14)
optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=0.2)
scheduler = CosineAnnealingLR(optimizer, T_max=32000)  # 32 epochs, batch=32768
epochs = 32
batch_size = 32768  # 超大 batch

# 数据增强 (比 SimCLR 简单)
transform = Compose([
    RandomResizedCrop(224),
    RandomHorizontalFlip(),
    ColorJitter(0.4, 0.4, 0.4),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

## 6. 下游应用

### 6.1 迁移学习

冻结 CLIP 编码器，添加线性分类头进行微调：

```python
class CLIPLinearProbe(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.encoder = clip_model.visual
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.head = nn.Linear(512, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.encoder(x)
        return self.head(features)
```

### 6.2 CLIP 作为损失函数

CLIP 可用于指导生成模型（如 Stable Diffusion）：

$$\mathcal{L}_{\text{CLIP}} = -\text{sim}(E_v(I), E_t(T))$$

## 7. CLIP 的局限性

- **细粒度理解不足**：难以区分细节差异（如计数、空间关系）
- **分布外泛化有限**：在专业领域（如医学影像）表现不佳
- **幻觉倾向**：对不存在的属性可能给出高置信度
- **数据偏见**：继承训练数据中的社会偏见

## 8. 小结

CLIP 开创了视觉-语言对比预训练的新范式，其零样本能力令人瞩目。
CLIP 的成功直接影响了后续的 ALIGN、BLIP、Stable Diffusion 等一系列工作，
成为多模态 AI 基础设施的核心组件。
