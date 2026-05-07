# BLIP / BLIP-2 多模态预训练

## 1. BLIP 简介

BLIP (Bootstrapping Language-Image Pre-training) 由 Salesforce 于 2022 年提出。
BLIP 的核心贡献是 **CapFilt 机制**——通过字幕生成和过滤来提升预训练数据质量。

BLIP 同时支持理解任务和生成任务，是第一代统一视觉-语言理解与生成的模型。

## 2. BLIP 架构

### 2.1 MED (Multimodal mixture of Encoder-Decoder)

BLIP 使用三个模型共用一套参数的架构：

```
                ┌─→ 文本编码器 (ITM, ITC)
输入图像 ─→ 视觉编码器 ─→ 多模态特征 ─┤─→ 图像-文本编码器 (ITM)
                └─→ 文本解码器 (图像描述)
```

```python
class BLIPModel(nn.Module):
    def __init__(self, vision_encoder, text_decoder, hidden_size=768):
        super().__init__()
        self.vision_encoder = vision_encoder  # ViT-L/16

        # 共享的 Cross-Attention 层
        self.cross_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 12, batch_first=True),
            num_layers=6
        )

        # 三种模式的头部
        self.itm_head = nn.Linear(hidden_size, 2)  # 图文匹配
        self.itc_proj = nn.Linear(hidden_size, 256)  # 图文对比
        self.text_decoder = text_decoder  # 文本生成

    def forward(self, images, texts, mode='itc'):
        vision_features = self.vision_encoder(images)

        if mode == 'itc':
            # 图文对比学习
            return self.itc_proj(vision_features)
        elif mode == 'itm':
            # 图文匹配 (二分类)
            fused = self.cross_attention(vision_features, texts)
            return self.itm_head(fused[:, 0, :])
        elif mode == 'caption':
            # 文本生成
            return self.text_decoder(texts, encoder_hidden_states=vision_features)
```

### 2.2 三种预训练任务

| 任务 | 目标 | 说明 |
|------|------|------|
| ITC (图文对比) | 对齐 | 与 CLIP 类似的对比学习 |
| ITM (图文匹配) | 理解 | 二分类判断图文是否匹配 |
| LM (语言建模) | 生成 | 以视觉特征为条件生成文本 |

## 3. CapFilt 机制

### 3.1 核心思想

CapFilt (Captioning and Filtering) 通过两步提升数据质量：

```
原始噪声数据
    ↓
Captioner (字幕生成器) → 为图像生成新描述
    ↓
Filter (过滤器) → 判断图文对是否匹配
    ↓
高质量数据
```

### 3.2 实现细节

```python
class CapFilt:
    def __init__(self, captioner, filter_model):
        self.captioner = captioner
        self.filter = filter_model

    def generate_captions(self, images, num_captions=3):
        """为每个图像生成多个候选描述"""
        captions = []
        for img in images:
            candidates = self.captioner.generate(
                img, num_return_sequences=num_captions,
                do_sample=True, top_p=0.9
            )
            captions.append(candidates)
        return captions

    def filter_captions(self, images, captions):
        """过滤不匹配的图文对"""
        filtered_pairs = []
        for img, caps in zip(images, captions):
            for cap in caps:
                itm_score = self.filter(img, cap)  # 匹配分数
                if itm_score > 0.5:  # 阈值
                    filtered_pairs.append((img, cap))
        return filtered_pairs
```

### 3.3 效果

| 数据来源 | ImageNet 零样本 |
|---------|---------------|
| 原始噪声数据 | 76.5% |
| + CapFilt | 78.3% |
| + CapFilt (Sythetic) | 79.3% |

## 4. BLIP-2 简介

BLIP-2 由 Salesforce 于 2023 年提出，核心创新是 **Q-Former 架构**，
用于桥接冻结的视觉编码器和冻结的大语言模型。

### 4.1 核心思想

```
冻结的视觉编码器 → Q-Former → 冻结的 LLM
   (CLIP ViT)    (可训练)   (Flan-T5/ OPT)
```

只训练 Q-Former，大幅降低训练成本。

## 5. Q-Former 架构

### 5.1 设计

Q-Former 是一个轻量级的 Transformer，包含两种注意力：

```
自注意力 (Self-Attention)  ← Query tokens 之间
交叉注意力 (Cross-Attention) ← Query tokens 与图像特征
```

```python
class QFormer(nn.Module):
    def __init__(self, num_queries=32, hidden_size=768, num_layers=12):
        super().__init__()
        # 可学习的查询 token
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))

        # Transformer 层 (包含自注意力和交叉注意力)
        self.layers = nn.ModuleList([
            QFormerLayer(hidden_size, num_heads=12)
            for _ in range(num_layers)
        ])

        # 输出投影到 LLM 的嵌入空间
        self.output_proj = nn.Linear(hidden_size, 2560)  # 投影到 T5 维度

    def forward(self, image_features, text_input=None):
        # image_features: (B, N, D) 来自冻结的视觉编码器
        queries = self.query_tokens.expand(image_features.size(0), -1, -1)

        for layer in self.layers:
            queries = layer(
                queries=queries,
                image_features=image_features,
                text_input=text_input
            )

        return self.output_proj(queries)


class QFormerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

    def forward(self, queries, image_features, text_input=None):
        # 自注意力
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]

        # 交叉注意力 (图像)
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, image_features, image_features)[0]

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries
```

### 5.2 预训练阶段

BLIP-2 分两阶段预训练：

**阶段一：视觉-语言表示学习**

使用图文对比 (ITC)、图文匹配 (ITM)、图像引导的文本生成 (ITG) 任务：

$$\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{ITC}} + \mathcal{L}_{\text{ITM}} + \mathcal{L}_{\text{ITG}}$$

**阶段二：视觉-语言生成学习**

将 Q-Former 的输出接入冻结的 LLM：

$$\mathcal{L}_{\text{stage2}} = -\sum_{t} \log P(x_t | x_{<t}, \mathbf{q}_{\text{visual}})$$

## 6. BLIP vs BLIP-2 对比

| 特性 | BLIP | BLIP-2 |
|------|------|--------|
| 视觉编码器 | 自训练 ViT | 冻结 CLIP ViT |
| 文本编码器 | 自训练 | 冻结 LLM |
| 桥接模块 | Cross-Attention | Q-Former |
| 可训练参数 | 全部 | 仅 Q-Former |
| 训练成本 | 高 | 低 |
| 多模态理解 | 较好 | 很强 |
| 对话能力 | 无 | 支持 |

## 7. 使用示例

```python
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# 图像问答
image = load_image("cat.jpg")
inputs = processor(images=image, text="Question: what is the cat doing?", return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))

# 图像描述
inputs = processor(images=image, return_tensors="pt")
out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True))
```

## 8. 小结

BLIP 通过 CapFilt 机制解决了噪声数据问题，BLIP-2 通过 Q-Former 实现了高效的
冻结模型桥接。BLIP-2 的设计思想（只训练桥接模块）深刻影响了后续的 LLaVA 等模型，
成为多模态大模型的标准范式之一。
