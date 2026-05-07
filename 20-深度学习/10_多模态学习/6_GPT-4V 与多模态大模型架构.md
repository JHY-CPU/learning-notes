# GPT-4V 与多模态大模型架构

## 1. GPT-4V 简介

GPT-4V (GPT-4 with Vision) 是 OpenAI 于 2023 年发布的多模态大语言模型，
能够同时理解和生成文本与图像。GPT-4V 的具体架构未公开发布，
但通过公开信息和研究社区的分析，我们可以理解其核心设计思路。

## 2. 视觉编码器 + LLM 集成方式

### 2.1 常见范式

多模态大模型将视觉信息接入语言模型的主流方式：

```
方案1: 视觉 Token 注入
    图像 → 视觉编码器 → 线性投影 → [视觉Token₁, ..., 视觉Tokenₙ] + [文本Token₁, ...] → LLM

方案2: 交叉注意力
    图像 → 视觉编码器 → 键值对
    文本 → LLM + 交叉注意力层(查询: 文本, 键值: 视觉)

方案3: Adapter/桥接模块
    图像 → 视觉编码器 → Q-Former/MLP → 适配Token → LLM
```

### 2.2 GPT-4V 推测架构

基于 OpenAI 的技术报告和社区分析，GPT-4V 可能采用方案1：

```python
class GPT4VArchitecture(nn.Module):
    """
    推测的 GPT-4V 架构 (基于公开信息)
    """
    def __init__(self, vision_encoder, llm, visual_token_num=576):
        super().__init__()
        self.vision_encoder = vision_encoder  # 可能为 ViT-L/14 或更大
        self.llm = llm  # GPT-4 (decoder-only Transformer)

        # 视觉-语言投影层
        self.visual_projection = nn.Sequential(
            nn.Linear(1024, 4096),  # ViT 特征 → LLM 隐层维度
            nn.GELU(),
            nn.Linear(4096, 4096)
        )

        # 特殊的视觉 token embedding
        self.visual_token_id = 32000  # 特殊 token

    def forward(self, images, text_tokens):
        # 1. 编码图像
        vision_features = self.vision_encoder(images)  # (B, N, 1024)
        visual_tokens = self.visual_projection(vision_features)  # (B, N, 4096)

        # 2. 编码文本
        text_embeddings = self.llm.token_embedding(text_tokens)

        # 3. 拼接: 视觉 token 放在文本前面
        combined = torch.cat([visual_tokens, text_embeddings], dim=1)

        # 4. LLM 前向传播
        output = self.llm(combined)
        return output
```

## 3. 视觉 Token 机制

### 3.1 图像 Patch 到 Token 的映射

图像被划分为固定大小的 patch，每个 patch 独立编码为一个视觉 token：

$$\mathbf{v}_i = \text{Proj}(\text{ViT}(\text{patch}_i)), \quad i = 1, \ldots, N$$

```python
class ImageToVisualTokens(nn.Module):
    def __init__(self, patch_size=14, embed_dim=1024, llm_dim=4096):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(embed_dim, llm_dim)

    def forward(self, image, max_tokens=576):
        """
        image: (B, 3, H, W)
        返回: (B, N, llm_dim) 视觉 token
        """
        # ViT 编码 (不含 cls token)
        features = self.vit_encoder(image)[:, 1:, :]  # (B, N, 1024)

        # 动态分辨率处理
        if features.size(1) > max_tokens:
            # 下采样: 空间维度池化
            h = w = int(features.size(1) ** 0.5)
            features = features.reshape(-1, h, w, 1024)
            features = F.adaptive_avg_pool2d(
                features.permute(0, 3, 1, 2),
                int(max_tokens ** 0.5)
            ).permute(0, 2, 3, 1).reshape(-1, max_tokens, 1024)

        return self.proj(features)
```

### 3.2 分辨率处理策略

| 策略 | 优点 | 缺点 | 代表模型 |
|------|------|------|---------|
| 固定分辨率 | 简单 | 细节丢失 | BLIP-2 |
| 动态切片 | 保留细节 | token 数量多 | LLaVA-1.5 |
| 分层编码 | 平衡效率和精度 | 实现复杂 | Fuyu |
| 像素混洗 | 高效压缩 | 可能丢失信息 | Qwen-VL |

## 4. 多模态大模型架构对比

| 模型 | 视觉编码器 | 桥接方式 | LLM | 训练策略 |
|------|----------|---------|-----|---------|
| GPT-4V | 未公开 | 未公开 | 未公开 | 未公开 |
| Gemini | 原生多模态 | 端到端 | Gemini | 端到端多模态 |
| LLaVA | CLIP ViT | 线性投影 | Vicuna | 两阶段训练 |
| Qwen-VL | ViT | 单层交叉注意力 | Qwen-7B | 三阶段训练 |
| InternVL | InternViT-6B | QLLaMA | InternLM | 三阶段训练 |

## 5. 视觉 Token 的挑战

### 5.1 Token 数量问题

高分辨率图像会生成大量视觉 token，严重影响推理效率：

$$\text{视觉 token 数} = \frac{H}{P} \times \frac{W}{P}$$

其中 $H, W$ 为图像尺寸，$P$ 为 patch 大小。

| 图像尺寸 | Patch=14 | Patch=16 |
|---------|----------|----------|
| 224x224 | 256 | 196 |
| 448x448 | 1024 | 784 |
| 1024x1024 | 5184 | 4096 |

### 5.2 解决方案

```python
# 方案1: Token 压缩 (Pixel Shuffle)
class PixelShuffleCompress(nn.Module):
    def __init__(self, scale=2, dim=1024):
        super().__init__()
        self.scale = scale
        self.proj = nn.Linear(dim * scale**2, dim)

    def forward(self, visual_tokens):
        """
        visual_tokens: (B, H*W, D)
        返回: (B, H*W/s², D) - 减少 s² 倍 token
        """
        B, N, D = visual_tokens.shape
        H = W = int(N ** 0.5)

        x = visual_tokens.reshape(B, H, W, D)
        x = x.reshape(B, H // self.scale, self.scale,
                      W // self.scale, self.scale, D)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(
            B, H // self.scale, W // self.scale, D * self.scale**2
        )
        return self.proj(x.reshape(B, -1, D * self.scale**2))

# 方案2: 动态 Token 选择
class DynamicTokenSelection(nn.Module):
    def __init__(self, dim, max_tokens=256):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)
        self.max_tokens = max_tokens

    def forward(self, visual_tokens):
        scores = self.scorer(visual_tokens).squeeze(-1)  # (B, N)
        _, indices = scores.topk(self.max_tokens, dim=1)
        selected = torch.gather(
            visual_tokens, 1,
            indices.unsqueeze(-1).expand(-1, -1, visual_tokens.size(-1))
        )
        return selected
```

## 6. GPT-4V 的能力边界

### 6.1 强项

- 复杂图像理解与推理
- 多图对比分析
- 文档、图表理解
- 创意视觉任务（梗图解释等）

### 6.2 弱项

- 精确的空间定位（如：左上角第三个像素）
- 细粒度计数（大数量物体）
- 专业领域图像（如放射科影像）
- 时间序列推理（视频）

## 7. 多模态大模型的推理流程

```python
@torch.no_grad()
def multimodal_inference(model, image, text_prompt, max_new_tokens=512):
    # 1. 图像编码
    visual_tokens = model.encode_image(image)

    # 2. 文本编码 + 视觉 Token 拼接
    text_tokens = model.tokenizer.encode(text_prompt)
    input_ids = torch.tensor([text_tokens])

    # 3. 构建输入序列
    visual_embeds = model.visual_projection(visual_tokens)
    text_embeds = model.llm.token_embedding(input_ids)

    # 特殊标记指示视觉 token 的位置
    inputs_embeds = torch.cat([visual_embeds, text_embeds], dim=1)

    # 4. 自回归生成
    for _ in range(max_new_tokens):
        outputs = model.llm(inputs_embeds=inputs_embeds)
        next_token = outputs.logits[:, -1, :].argmax(dim=-1)
        if next_token.item() == model.eos_token_id:
            break
        inputs_embeds = torch.cat([
            inputs_embeds,
            model.llm.token_embedding(next_token).unsqueeze(1)
        ], dim=1)

    return model.tokenizer.decode(inputs_embeds[0, visual_embeds.size(1):])
```

## 8. 小结

GPT-4V 代表了多模态大模型的最高水平，虽然其具体架构未公开，
但视觉 Token 注入 + LLM 的范式已成为主流。核心挑战在于如何高效地
将高分辨率视觉信息压缩为有限的 token，同时保留细粒度的空间和语义信息。
后续的 LLaVA、Qwen-VL 等开源模型为我们提供了更详细的架构细节。
