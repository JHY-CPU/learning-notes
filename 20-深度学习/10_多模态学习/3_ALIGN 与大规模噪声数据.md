# ALIGN 与大规模噪声数据

## 1. ALIGN 简介

ALIGN (A Larger-scale ImagiNg with noisy text embedding) 由 Google 于 2021 年提出。
与 CLIP 使用精心过滤的 4 亿数据不同，ALIGN 直接使用 **18 亿** 噪声图文对进行训练，
证明了**数据规模可以弥补数据质量**。

核心发现：**简单过滤 + 超大规模数据 = 强大的视觉-语言表示**。

## 2. 数据策略：噪声容忍

### 2.1 数据来源

ALIGN 使用 Google 内部的图文数据集，包含 18 亿 (image, alt-text) 对。
数据来源于互联网图片的 alt-text，天然包含大量噪声：

- 不相关的图文对
- alt-text 过于简短或不描述内容
- 重复数据
- 低质量图片

### 2.2 简单过滤策略

与 CLIP 的严格数据清洗不同，ALIGN 只做了极简的过滤：

```
1. 从训练集中去除超过 1000 个训练样本的重复文本
2. 子采样频繁出现的文本
3. 对文本执行基本清洗（去特殊字符等）
```

没有使用任何复杂的语义匹配或质量评估模型。

## 3. 模型架构

ALIGN 的双塔架构与 CLIP 非常相似：

### 3.1 视觉编码器

使用 EfficientNet-L2，输入分辨率 289x289：

| 模型 | 参数量 | 输入大小 | ImageNet Top-1 |
|------|--------|---------|---------------|
| EfficientNet-B7 | 66M | 600 | - |
| EfficientNet-L2 | 480M | 475 | 88.4% |

### 3.2 文本编码器

使用 BERT-large：

| 模型 | 参数量 | 层数 |
|------|--------|------|
| BERT-base | 110M | 12 |
| BERT-large | 340M | 24 |

```python
class ALIGNModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 视觉: EfficientNet
        self.image_encoder = EfficientNet.from_pretrained('efficientnet-l2')
        self.image_proj = nn.Linear(1280, 640)  # 投影到共享空间

        # 文本: BERT-large
        self.text_encoder = BertModel.from_pretrained('bert-large-uncased')
        self.text_proj = nn.Linear(1024, 640)

        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode_image(self, images):
        features = self.image_encoder(images)
        features = self.image_proj(features)
        return F.normalize(features, dim=-1)

    def encode_text(self, input_ids, attention_mask):
        outputs = self.text_encoder(input_ids, attention_mask=attention_mask)
        # 使用 [CLS] token
        features = self.text_proj(outputs.last_hidden_state[:, 0, :])
        return F.normalize(features, dim=-1)

    def forward(self, images, input_ids, attention_mask):
        img_features = self.encode_image(images)
        txt_features = self.encode_text(input_ids, attention_mask)

        # 对称对比损失
        logits = img_features @ txt_features.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)

        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss
```

## 4. 与 CLIP 的对比

| 特性 | CLIP | ALIGN |
|------|------|-------|
| 数据量 | 4 亿 | 18 亿 |
| 数据质量 | 高（严格过滤） | 低（简单过滤） |
| 视觉编码器 | ViT / ResNet | EfficientNet-L2 |
| 文本编码器 | Transformer (自训练) | BERT-large |
| 训练 batch size | 32,768 | 16,384 |
| ImageNet 零样本 | 76.2% | 76.4% |

### 4.1 关键发现

```python
# CLIP 的数据策略 (严格)
def clip_data_filtering(caption, image):
    # 使用精心设计的过滤规则
    if len(caption.split()) < 5:
        return False
    if is_duplicate(caption, image):
        return False
    if not semantic_match(caption, image, threshold=0.7):
        return False
    return True

# ALIGN 的数据策略 (宽松)
def align_data_filtering(caption, image):
    # 只做最基本的去重
    if caption in frequent_texts:
        return False  # 子采样
    return True
```

## 5. 噪声鲁棒性分析

### 5.1 为什么噪声数据也能工作？

1. **噪声是随机的**：随机噪声不会引入系统性偏差，大量数据下的信号仍然足够
2. **对比学习的特性**：负样本不需要绝对准确，batch 内的随机配对仍有统计意义
3. **规模效应**：18 亿数据即使 20% 噪声，有效数据仍有 14.4 亿

### 5.2 实验验证

| 噪声比例 | CLIP (400M 清洗) | ALIGN (1.8B 噪声) |
|---------|-----------------|-------------------|
| 0% (理想) | 75.3% | - |
| ~10% | - | 76.4% |
| ~20% | - | 75.8% |
| ~50% | 65.2% | 72.1% |

## 6. 训练过程

```python
# 训练配置
config = {
    "batch_size": 16384,
    "learning_rate": 0.001,  # 使用 SGD + momentum
    "momentum": 0.9,
    "weight_decay": 1e-6,
    "epochs": 1000000,  # steps
    "temperature_init": 0.07,
    "image_size": 289,
    "max_text_length": 64,
}

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=config["learning_rate"],
    momentum=config["momentum"],
    weight_decay=config["weight_decay"]
)

scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"])
```

## 7. 下游性能

ALIGN 在多个基准上与 CLIP 持平或超越：

| 任务 | CLIP ViT-L/14 | ALIGN |
|------|--------------|-------|
| ImageNet 零样本 | 75.3% | 76.4% |
| Flickr30K 检索 | 88.0% | 84.9% |
| COCO 检索 | 58.4% | 59.9% |
| 零样本分类 (avg) | 74.2% | 71.8% |

## 8. 启示

ALIGN 的核心启示：

1. **数据规模 > 数据质量**：在足够大的规模下，简单过滤就足够了
2. **预训练模型的价值**：使用 BERT 等预训练文本编码器可以加速收敛
3. **噪声容忍度**：对比学习天然对噪声有一定容忍能力

## 9. 小结

ALIGN 证明了在多模态预训练中，数据规模和噪声容忍可以替代精细的数据清洗。
这一发现启发了后续的 LAION 系列数据集和更大规模的视觉-语言模型训练，
为开源社区复现 CLIP 级别的模型铺平了道路。
