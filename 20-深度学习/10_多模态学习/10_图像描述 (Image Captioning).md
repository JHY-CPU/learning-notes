# 图像描述 (Image Captioning)

## 1. 任务定义

图像描述 (Image Captioning) 是让模型自动生成自然语言描述来描述图像内容。
给定图像 $I$，生成描述序列 $S = \{w_1, w_2, \ldots, w_T\}$，使得 $P(S|I)$ 最大化。

$$S^* = \arg\max_S P(S|I) = \arg\max_S \prod_{t=1}^{T} P(w_t | w_{<t}, I)$$

## 2. 编码器-解码器架构

### 2.1 基本框架

```
图像 → 视觉编码器 (CNN/ViT) → 视觉特征 → 解码器 (LSTM/Transformer) → 文本描述
```

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, vocab_size, embed_dim=512):
        super().__init__()
        self.encoder = encoder  # 预训练视觉编码器
        self.decoder = decoder  # 自回归解码器
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.output_proj = nn.Linear(embed_dim, vocab_size)

    def forward(self, images, captions):
        """
        images: (B, 3, H, W)
        captions: (B, T) token ids
        """
        # 视觉编码
        visual_features = self.encoder(images)  # (B, N, D)

        # 文本嵌入
        text_embeds = self.embed(captions)  # (B, T, D)

        # 解码
        decoder_out = self.decoder(text_embeds, encoder_hidden_states=visual_features)

        return self.output_proj(decoder_out)
```

### 2.2 CNN + LSTM (经典方案)

```python
class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        resnet = models.resnet101(pretrained=True)
        # 去掉最后的全连接和池化
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))  # 固定空间大小
        self.proj = nn.Linear(2048, embed_dim)

    def forward(self, images):
        features = self.backbone(images)  # (B, 2048, 7, 7)
        features = self.adaptive_pool(features)  # (B, 2048, 14, 14)
        features = features.flatten(2).transpose(1, 2)  # (B, 196, 2048)
        return self.proj(features)  # (B, 196, embed_dim)


class LSTMDecoder(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512, vocab_size=10000):
        super().__init__()
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.vocab_proj = nn.Linear(hidden_dim, vocab_size)

    def forward(self, text_embeds, visual_features, hidden=None):
        # LSTM 解码
        lstm_out, hidden = self.lstm(text_embeds, hidden)

        # 注意力
        attn_out, _ = self.attention(lstm_out, visual_features, visual_features)

        return self.vocab_proj(attn_out), hidden
```

## 3. 注意力机制

### 3.1 加性注意力 (Bahdanau Attention)

$$e_{ti} = v_a^\top \tanh(W_a h_{t-1} + U_a f_i)$$
$$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k} \exp(e_{tk})}$$
$$\hat{f}_t = \sum_i \alpha_{ti} f_i$$

### 3.2 乘性注意力 (Luong Attention)

$$e_{ti} = h_t^\top W_a f_i$$

### 3.3 空间注意力实现

```python
class SpatialAttention(nn.Module):
    def __init__(self, hidden_dim, visual_dim):
        super().__init__()
        self.W_h = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(visual_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_state, visual_features):
        """
        hidden_state: (B, D) 解码器当前隐藏状态
        visual_features: (B, N, V) 视觉特征
        """
        # 计算注意力分数
        h = self.W_h(hidden_state).unsqueeze(1)  # (B, 1, D)
        v = self.W_v(visual_features)  # (B, N, D)

        scores = self.v(torch.tanh(h + v)).squeeze(-1)  # (B, N)
        weights = F.softmax(scores, dim=-1)  # (B, N)

        # 加权求和
        context = (weights.unsqueeze(-1) * visual_features).sum(dim=1)  # (B, V)
        return context, weights
```

## 4. Transformer 解码器

### 4.1 带视觉编码的 Transformer

```python
class TransformerCaptionDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, captions, visual_features, tgt_mask=None):
        """
        captions: (B, T)
        visual_features: (B, N, D)
        """
        positions = torch.arange(captions.size(1), device=captions.device).unsqueeze(0)
        x = self.embed(captions) + self.pos_embed(positions)

        if tgt_mask is None:
            tgt_mask = self.generate_square_mask(captions.size(1)).to(captions.device)

        output = self.decoder(x, visual_features, tgt_mask=tgt_mask)
        return self.output_proj(output)

    def generate_square_mask(self, sz):
        return torch.triu(torch.ones(sz, sz), diagonal=1).bool()
```

## 5. 评估指标

### 5.1 BLEU (Bilingual Evaluation Understudy)

$$\text{BLEU}_n = \text{BP} \cdot \exp\left(\sum_{n=1}^{N} w_n \log p_n\right)$$

其中 $p_n$ 为 n-gram 精确率。

### 5.2 CIDEr (Consensus-based Image Description Evaluation)

CIDEr 专门针对图像描述设计，使用 TF-IDF 加权的 n-gram 匹配：

$$\text{CIDEr}_n = \frac{1}{M} \sum_i \frac{\mathbf{g}_i^n \cdot \mathbf{g}_{S_{ij}}^n}{\|\mathbf{g}_i^n\| \cdot \|\mathbf{g}_{S_{ij}}^n\|}$$

### 5.3 METEOR 和 ROUGE

```python
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor

def evaluate_captioning(predictions, references):
    """
    predictions: {image_id: [predicted_caption]}
    references: {image_id: [ref1, ref2, ...]}
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Cider(), "CIDEr"),
    ]

    results = {}
    for scorer, method in scorers:
        score, _ = scorer.compute_score(references, predictions)
        if isinstance(method, list):
            for s, m in zip(score, method):
                results[m] = s
        else:
            results[method] = score

    return results
```

### 5.4 指标对比

| 指标 | 侧重点 | 范围 | 说明 |
|------|--------|------|------|
| BLEU | n-gram 精确率 | 0-1 | 机器翻译常用，粒度粗糙 |
| METEOR | 精确率+召回率 | 0-1 | 考虑同义词 |
| CIDEr | 共识性 | 0-10+ | 专为图像描述设计 |
| ROUGE | n-gram 召回率 | 0-1 | 摘要任务常用 |
| SPICE | 语义命题 | 0-1 | 场景图匹配 |

## 6. 采样策略

### 6.1 贪心解码

$$w_t = \arg\max_{w} P(w | w_{<t}, I)$$

### 6.2 束搜索 (Beam Search)

```python
def beam_search(model, image, beam_width=5, max_len=50):
    visual_features = model.encoder(image)

    # 初始: <BOS>
    sequences = [([model.bos_idx], 0.0)]  # (tokens, log_prob)

    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if seq[-1] == model.eos_idx:
                all_candidates.append((seq, score))
                continue

            input_tensor = torch.tensor([seq])
            logits = model.decoder(input_tensor, visual_features)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
            topk_probs, topk_ids = log_probs.topk(beam_width)

            for prob, idx in zip(topk_probs[0], topk_ids[0]):
                candidate = (seq + [idx.item()], score + prob.item())
                all_candidates.append(candidate)

        # 保留 top-k
        sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    return sequences[0][0]  # 最高分序列
```

### 6.3 核采样 (Nucleus Sampling)

```python
def nucleus_sampling(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # 移除累积概率超过 p 的 token
    remove_mask = cumulative_probs > p
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False

    sorted_logits[remove_mask] = float('-inf')
    return F.softmax(sorted_logits, dim=-1)
```

## 7. 基准性能

| 模型 | BLEU-4 | METEOR | CIDEr | SPICE |
|------|--------|--------|-------|-------|
| Show & Tell (2015) | 27.7 | 23.7 | 85.5 | - |
| Bottom-Up Top-Down | 36.2 | 27.0 | 120.1 | 21.5 |
| OSCAR | 40.5 | 30.6 | 140.0 | 24.5 |
| BLIP | 40.4 | 33.2 | 136.7 | 24.2 |
| BLIP-2 | 43.7 | 34.0 | 145.8 | 26.0 |

## 8. 小结

图像描述从早期的 CNN+LSTM 到现代的 Transformer 架构，经历了巨大的发展。
注意力机制、预训练视觉编码器和大语言模型的引入显著提升了生成质量。
CIDEr 等专用指标为评估图像描述质量提供了可靠标准。
现代模型如 BLIP-2 已经能够生成非常自然和准确的图像描述。
