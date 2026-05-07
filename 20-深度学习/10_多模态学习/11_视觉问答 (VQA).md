# 视觉问答 (VQA)

## 1. 任务定义

视觉问答 (Visual Question Answering, VQA) 要求模型根据图像内容回答自然语言问题。
给定图像 $I$ 和问题 $Q$，模型需要生成答案 $A$：

$$P(A|I, Q) = f(\text{Encoder}_v(I), \text{Encoder}_t(Q))$$

VQA 是检验多模态理解能力的核心任务，要求模型同时具备视觉感知和语言理解能力。

## 2. 问题类型

| 类型 | 示例 | 难度 |
|------|------|------|
| 是非问题 | "图中有猫吗?" | 简单 |
| 计数问题 | "图中有几个苹果?" | 中等 |
| 颜色问题 | "猫是什么颜色的?" | 简单 |
| 位置问题 | "球在哪里?" | 中等 |
| 推理问题 | "这个人为什么在跑?" | 困难 |
| 常识问题 | "这是什么季节?" | 困难 |

## 3. 特征融合方式

### 3.1 逐元素乘法融合

最简单的融合方式，逐元素相乘：

$$\mathbf{h} = \text{MLP}(\mathbf{v} \odot \mathbf{q})$$

```python
class ElementWiseFusion(nn.Module):
    def __init__(self, visual_dim, question_dim, hidden_dim):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.q_proj = nn.Linear(question_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, visual_feat, question_feat):
        v = self.v_proj(visual_feat)
        q = self.q_proj(question_feat)
        return self.fusion(v * q)  # 逐元素乘法
```

### 3.2 双线性融合 (MCB)

$$\mathbf{h} = \Phi(\mathbf{v})^\top \Phi(\mathbf{q})$$

使用随机投影近似双线性池化：

```python
class MultimodalCompactBilinearPooling(nn.Module):
    def __init__(self, visual_dim, question_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        # 随机投影矩阵 (固定)
        self.h_v = nn.Parameter(torch.randint(0, 2, (visual_dim,)) * 2 - 1, requires_grad=False)
        self.h_q = nn.Parameter(torch.randint(0, 2, (question_dim,)) * 2 - 1, requires_grad=False)
        self.s_v = nn.Parameter(torch.randint(0, 2, (visual_dim,)) * 2 - 1, requires_grad=False)
        self.s_q = nn.Parameter(torch.randint(0, 2, (question_dim,)) * 2 - 1, requires_grad=False)

    def count_sketch(self, x, h, s):
        """Count Sketch 散列"""
        batch_size = x.size(0)
        sketch = torch.zeros(batch_size, self.output_dim, device=x.device)
        for i in range(x.size(-1)):
            sketch[:, h[i]] += s[i] * x[:, i]
        return sketch

    def forward(self, visual_feat, question_feat):
        sketch_v = self.count_sketch(visual_feat, self.h_v, self.s_v)
        sketch_q = self.count_sketch(question_feat, self.h_q, self.s_q)

        # FFT 进行卷积 = 频域乘法
        fft_v = torch.fft.fft(sketch_v, dim=-1)
        fft_q = torch.fft.fft(sketch_q, dim=-1)
        return torch.fft.ifft(fft_v * fft_q, dim=-1).real
```

### 3.3 门控融合

$$\mathbf{g} = \sigma(W_g[\mathbf{v}; \mathbf{q}])$$
$$\mathbf{h} = \mathbf{g} \odot \tanh(W_v \mathbf{v} + W_q \mathbf{q})$$

```python
class GatedFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        self.transform_v = nn.Linear(dim, dim)
        self.transform_q = nn.Linear(dim, dim)

    def forward(self, v, q):
        g = self.gate(torch.cat([v, q], dim=-1))
        h = torch.tanh(self.transform_v(v) + self.transform_q(q))
        return g * h
```

## 4. 注意力机制

### 4.1 问题引导的视觉注意力

```python
class QuestionGuidedAttention(nn.Module):
    def __init__(self, visual_dim, question_dim, hidden_dim=256):
        super().__init__()
        self.v_proj = nn.Linear(visual_dim, hidden_dim)
        self.q_proj = nn.Linear(question_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, visual_features, question_feature):
        """
        visual_features: (B, N, D_v) - N 个图像区域
        question_feature: (B, D_q)
        """
        v = self.v_proj(visual_features)  # (B, N, H)
        q = self.q_proj(question_feature).unsqueeze(1)  # (B, 1, H)

        scores = self.attn(torch.tanh(v + q)).squeeze(-1)  # (B, N)
        weights = F.softmax(scores, dim=-1)  # (B, N)

        # 加权视觉特征
        context = (weights.unsqueeze(-1) * visual_features).sum(dim=1)  # (B, D_v)
        return context, weights
```

### 4.2 多步推理 (Stacked Attention)

```python
class StackedAttention(nn.Module):
    def __init__(self, num_steps=2, visual_dim=2048, question_dim=1024):
        super().__init__()
        self.num_steps = num_steps
        self.attention_layers = nn.ModuleList([
            QuestionGuidedAttention(visual_dim, question_dim)
            for _ in range(num_steps)
        ])
        self.query_proj = nn.Linear(question_dim + visual_dim, question_dim)

    def forward(self, visual_features, question):
        query = question

        for i in range(self.num_steps):
            context, weights = self.attention_layers[i](visual_features, query)
            # 更新查询: 拼接上下文和当前查询
            query = self.query_proj(torch.cat([query, context], dim=-1))

        return query
```

## 5. 端到端 VQA 模型

```python
class VQAModel(nn.Module):
    def __init__(self, vision_encoder, text_encoder, num_answers=3129, hidden_dim=1024):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

        # 注意力融合
        self.visual_attention = QuestionGuidedAttention(2048, 1024)

        # 答案分类
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_answers)
        )

    def forward(self, images, questions, question_lengths):
        # 视觉编码
        visual_features = self.vision_encoder(images)  # (B, N, 2048)

        # 问题编码
        question_embeds = self.text_encoder(questions)  # (B, L, 1024)
        question_feature = question_embeds[:, 0, :]  # [CLS] token

        # 注意力加权
        visual_context, attn_weights = self.visual_attention(visual_features, question_feature)

        # 融合 & 分类
        fused = torch.cat([visual_context, question_feature], dim=-1)
        logits = self.classifier(fused)

        return logits, attn_weights
```

## 6. 数据集

### 6.1 VQA v2

| 统计 | 数值 |
|------|------|
| 图像数量 | 204,721 (MS-COCO) |
| 问题数量 | 1,100,000+ |
| 答案数量 | 3,129 个最常见答案 |
| 分割 | 训练 82K / 验证 40K / 测试 81K |

### 6.2 数据格式

```json
{
    "question_id": 262148000,
    "image_id": 262148,
    "question": "Where is he looking?",
    "answers": [
        {"answer": "down", "answer_confidence": "yes"},
        {"answer": "down", "answer_confidence": "yes"},
        {"answer": "at table", "answer_confidence": "maybe"}
    ],
    "answer_type": "other"
}
```

### 6.3 评估指标

$$\text{VQA Accuracy} = \min\left(\frac{\text{\# humans that provided that answer}}{3}, 1\right)$$

## 7. 基准性能

| 模型 | VQA v2 test-dev | VQA v2 test-std |
|------|----------------|----------------|
| Bottom-Up Top-Down | 70.3% | 70.3% |
| Pythia | 70.0% | 70.2% |
| OSCAR | 73.6% | 73.8% |
| BLIP | 78.3% | 78.3% |
| BLIP-2 | 82.9% | 82.6% |
| LLaVA-1.5 | 80.0% | - |

## 8. 挑战与前沿

### 8.1 常见问题

- **语言先验**：模型倾向于利用语言统计偏差而非视觉信息
- **视觉接地不足**：无法准确定位问题相关的视觉区域
- **组合推理困难**：需要多步推理的问题表现差

### 8.2 开放式 VQA

传统 VQA 是封闭集分类（从固定答案集选择），开放式 VQA 允许自由文本回答：

```python
# 传统 VQA (分类)
logits = model(image, question)  # (B, num_answers)
answer = answer_vocab[logits.argmax()]

# 开放式 VQA (生成)
answer = model.generate(image, question)  # 任意文本
```

## 9. 小结

VQA 是多模态理解的核心任务，要求模型在视觉和语言之间建立深层语义联系。
特征融合（逐元素、双线性、门控）和注意力机制是 VQA 的关键技术。
现代的 VQA 系统（如 BLIP-2）已经接近人类水平，但在复杂推理和语言先验问题上仍有提升空间。
