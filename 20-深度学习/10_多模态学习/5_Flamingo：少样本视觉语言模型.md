# Flamingo：少样本视觉语言模型

## 1. Flamingo 简介

Flamingo 由 DeepMind 于 2022 年提出，是第一个能够在**少样本**设置下
完成视觉-语言任务的模型。Flamingo 只需提供少量示例 (in-context examples)，
即可快速适应新任务，类似于 GPT-3 在纯文本领域的 few-shot 能力。

核心创新：**Perceiver Resampler** + **交错图文输入** + **门控交叉注意力层**。

## 2. 整体架构

```
交错输入: [Image₁ Text₁ Image₂ Text₂ ...]
    ↓
视觉编码器 (冻结 CLIP) → 视觉特征
    ↓
Perceiver Resampler → 固定数量的视觉 token
    ↓
冻结的 LLM + 门控交叉注意力层 → 文本输出
```

## 3. Perceiver Resampler

### 3.1 动机

视觉编码器输出的 token 数量取决于输入图像大小，导致序列长度不可控。
Perceiver Resampler 将可变长度的视觉特征压缩为固定长度的 token。

### 3.2 实现

```python
class PerceiverResampler(nn.Module):
    def __init__(self, dim=1024, num_latents=64, depth=6, num_heads=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_latents, dim))

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'cross_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'self_attn': nn.MultiheadAttention(dim, num_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'norm1': nn.LayerNorm(dim),
                'norm2': nn.LayerNorm(dim),
                'norm3': nn.LayerNorm(dim),
            })
            for _ in range(depth)
        ])

    def forward(self, visual_features):
        """
        visual_features: (B, N, D) 可变长度的视觉 token
        返回: (B, num_latents, D) 固定长度的视觉 token
        """
        latents = self.latents.expand(visual_features.size(0), -1, -1)

        for layer in self.layers:
            # 交叉注意力: latents attend to visual features
            lat = layer['norm1'](latents)
            latents = latents + layer['cross_attn'](lat, visual_features, visual_features)[0]

            # 自注意力
            lat = layer['norm2'](latents)
            latents = latents + layer['self_attn'](lat, lat, lat)[0]

            # FFN
            latents = latents + layer['ffn'](layer['norm3'](latents))

        return latents  # (B, 64, D)
```

### 3.3 压缩效果

| 输入视觉 token | Perceiver 输出 | 压缩比 |
|--------------|---------------|--------|
| 1024 (28x28 patches) | 64 | 16x |
| 256 (16x16 patches) | 64 | 4x |
| 576 (24x24 patches) | 64 | 9x |

## 4. 交错图文输入

Flamingo 处理**交错的视觉-文本输入**，即在一个序列中混合图像和文本：

```
x₁ = <image> A cat is sitting on a mat. <image> A dog is running in the park. ...
```

### 4.1 门控交叉注意力层

在冻结的 LLM 每个 Transformer 层中插入可训练的交叉注意力层：

$$\hat{h}^l = h^l + \text{tanh}(\alpha) \cdot \text{CrossAttn}(\text{LN}(h^l), v)$$

其中 $\alpha$ 为可学习门控参数，初始化为 0（训练开始时相当于原始 LLM）。

```python
class GatedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=16):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

        # 门控参数，初始化为 0
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, text_features, visual_tokens):
        """
        text_features: (B, L, D) 文本隐藏状态
        visual_tokens: (B, V, D) Perceiver 输出的视觉 token
        """
        residual = text_features
        text_features = self.norm(text_features)

        attn_out, _ = self.cross_attn(text_features, visual_tokens, visual_tokens)

        # 门控残差连接
        return residual + torch.tanh(self.alpha) * attn_out
```

### 4.2 交错处理流程

```python
class FlamingoBlock(nn.Module):
    def __init__(self, frozen_lm_layer, dim, num_heads=16):
        super().__init__()
        self.lm_layer = frozen_lm_layer  # 冻结
        self.gated_cross_attn = GatedCrossAttention(dim, num_heads)

    def forward(self, hidden_states, visual_tokens, text_positions):
        """
        text_positions: 标记哪些位置是文本 token
        """
        # 先做交叉注意力 (只在文本位置)
        text_mask = text_positions.unsqueeze(-1).float()
        cross_out = self.gated_cross_attn(hidden_states, visual_tokens)
        hidden_states = hidden_states * (1 - text_mask) + cross_out * text_mask

        # 再做正常的自注意力 + FFN
        hidden_states = self.lm_layer(hidden_states)
        return hidden_states
```

## 5. 训练策略

### 5.1 预训练数据

使用大规模图文交错数据集 M3W (MultiModal Massive-Web)：
- 从互联网网页中提取交错的图文内容
- 43M 网页，1.8B 图像

### 5.2 训练损失

$$\mathcal{L} = -\sum_{t} \log P(x_t | x_{<t}, \mathbf{v}_{\leq t})$$

即在给定前面的文本和对应位置的图像条件下，自回归地预测下一个 token。

### 5.3 冻结策略

| 组件 | 是否训练 | 原因 |
|------|---------|------|
| 视觉编码器 | 冻结 | 保留预训练视觉表示 |
| LLM 主体 | 冻结 | 保留语言能力 |
| Perceiver Resampler | 训练 | 桥接视觉和语言 |
| 门控交叉注意力 | 训练 | 注入视觉信息 |

## 6. 少样本能力

### 6.1 In-Context Learning

```
<图像1> "这是猫"
<图像2> "这是狗"
<图像3> "这是鸟"
<测试图像> "这是___"  → 模型生成 "这是大象"
```

### 6.2 性能提升

| 小样本数 | Flamingo-80B | 最佳专用模型 |
|---------|-------------|------------|
| 0-shot | 52.0% | - |
| 4-shot | 56.3% | - |
| 8-shot | 58.0% | - |
| 32-shot | 60.2% | 57.1% (微调) |

## 7. 模型规模

| 模型 | LLM 参数 | 总参数 | 视觉编码器 |
|------|---------|--------|----------|
| Flamingo-3B | 1.4B | 3.2B | NFNet-F6 |
| Flamingo-9B | 7.1B | 9.3B | NFNet-F6 |
| Flamingo-80B | 70B | 80B | NFNet-F6 |

## 8. 与后续工作的关系

Flamingo 的设计思想影响了多个后续工作：

- **LLaVA**：简化了视觉-LLM 桥接方式
- **InstructBLIP**：结合了指令微调
- **Otter**：改进了 in-context learning

## 9. 局限性

- 训练成本极高（80B 模型需要数千 GPU 小时）
- 少样本学习对示例质量敏感
- 难以处理复杂的空间推理
- 视觉 token 数量有限（64个），信息损失较大

## 10. 小结

Flamingo 开创了视觉-语言 in-context learning 的范式，证明了冻结的预训练模型
可以通过轻量级的桥接模块实现强大的多模态能力。Perceiver Resampler 和门控交叉注意力
的设计成为后续多模态大模型的重要参考。
