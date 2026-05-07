# LLaVA：视觉指令微调

## 1. LLaVA 简介

LLaVA (Large Language and Vision Assistant) 由威斯康星大学于 2023 年提出。
LLaVA 的核心创新是用**视觉指令微调 (Visual Instruction Tuning)** 将语言模型的
指令跟随能力迁移到多模态领域。LLaVA 是首个开源的端到端多模态对话模型。

核心公式：**冻结的 CLIP ViT + 线性投影 + 冻结的 Vicuna = 多模态助手**

## 2. 模型架构

### 2.1 整体设计

LLaVA 的架构极其简洁：

```
图像 → CLIP ViT-L/14 → 线性投影 → 视觉 Token
                                    ↓
用户文本 → Token Embedding → [视觉Token + 文本Token] → Vicuna → 回复
```

```python
class LLaVAModel(nn.Module):
    def __init__(self, vision_encoder, llm, vision_hidden=1024, llm_hidden=4096):
        super().__init__()
        self.vision_encoder = vision_encoder  # 冻结的 CLIP ViT-L/14
        self.llm = llm  # 冻结的 Vicuna

        # 可训练的投影层 (简单的线性映射)
        self.projection = nn.Linear(vision_hidden, llm_hidden)

    def encode_image(self, images):
        """编码图像为视觉 token"""
        with torch.no_grad():
            # CLIP ViT 输出，去掉 CLS token
            vision_features = self.vision_encoder(images)[:, 1:, :]
        # 投影到 LLM 嵌入空间
        return self.projection(vision_features)

    def forward(self, images, input_ids, labels=None):
        # 1. 图像编码
        visual_tokens = self.encode_image(images)  # (B, 256, 4096)

        # 2. 文本嵌入
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, L, 4096)

        # 3. 拼接
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        # 4. LLM 前向传播
        if labels is not None:
            # 补充 labels 维度以匹配拼接后的序列
            visual_labels = torch.full(
                (visual_tokens.size(0), visual_tokens.size(1)),
                -100, dtype=torch.long, device=labels.device
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return self.llm(inputs_embeds=inputs_embeds, labels=labels)
```

### 2.2 关键设计选择

| 组件 | LLaVA-1 | LLaVA-1.5 | LLaVA-NeXT |
|------|---------|-----------|------------|
| 视觉编码器 | CLIP ViT-L/14 | CLIP ViT-L/14@336 | SigLIP SO400M |
| 投影层 | 线性层 | 2层MLP | 2层MLP |
| LLM | Vicuna-13B | Vicuna-13B | Qwen-72B |
| 图像分辨率 | 224x224 | 336x336 | 动态切片 |
| 可训练参数 | 仅投影层 | 仅投影层 | 投影层+LLM |

## 3. 视觉指令数据构建

### 3.1 GPT-Assisted 数据生成

LLaVA 使用 GPT-4 自动生成视觉指令数据：

```python
# 使用 GPT-4 生成指令数据的流程
instruction_template = """
给定以下图像描述和边界框信息:
描述: {captions}
边界框: {bboxes}

请用以下三种方式与用户对话:
1. 对话: 提出关于图像的自然问题
2. 详细描述: 详细描述图像内容
3. 复杂推理: 提出需要推理的问题
"""

def generate_instruction_data(image_metadata, gpt4_api):
    """使用 GPT-4 生成视觉指令数据"""
    conversations = []

    for item in image_metadata:
        prompt = instruction_template.format(
            captions=item['captions'],
            bboxes=item['bboxes']
        )

        response = gpt4_api.chat(prompt)

        # 解析为多轮对话格式
        conv = parse_conversation(response)
        conversations.append(conv)

    return conversations
```

### 3.2 数据格式

```json
{
    "id": "00000001",
    "image": "coco/train2017/00000001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n这张图片里有什么?"
        },
        {
            "from": "gpt",
            "value": "图片中有一只猫正趴在窗台上，窗外可以看到蓝天和白云。"
        },
        {
            "from": "human",
            "value": "猫是什么颜色的?"
        },
        {
            "from": "gpt",
            "value": "这是一只橘色的猫，毛色温暖明亮。"
        }
    ]
}
```

### 3.3 数据规模

| 数据集 | 对话数量 | 图像数量 |
|--------|---------|---------|
| LLaVA-Instruct-150K | 150K | 80K (COCO) |
| LLaVA-NeXT | 760K | 混合多源 |
| ShareGPT4V | 100K | 100K |

## 4. 训练流程

### 4.1 两阶段训练

**阶段一：特征对齐预训练**

仅训练投影层，使用 CC-595K (Conceptual Captions) 数据：

```python
# 阶段一: 冻结视觉编码器和LLM，只训练投影层
optimizer_stage1 = AdamW(model.projection.parameters(), lr=1e-3)
for epoch in range(1):
    for images, captions in cc595k_dataloader:
        loss = model(images, tokenize(captions), labels=tokenize(captions))
        loss.backward()
        optimizer.step()
```

**阶段二：视觉指令微调**

继续训练投影层（可选微调 LLM），使用 LLaVA-Instruct-150K：

```python
# 阶段二: 训练投影层 + 可选微调LLM
trainable_params = list(model.projection.parameters())
if fine_tune_llm:
    trainable_params += get_lora_params(model.llm)

optimizer_stage2 = AdamW(trainable_params, lr=2e-5)
for epoch in range(3):
    for batch in instruction_dataloader:
        loss = model(**batch)
        loss.backward()
        optimizer.step()
```

### 4.2 训练配置

```python
training_config_stage1 = {
    "batch_size": 128,
    "learning_rate": 1e-3,
    "epochs": 1,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
}

training_config_stage2 = {
    "batch_size": 32,
    "learning_rate": 2e-5,
    "epochs": 3,
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "lora_rank": 128,  # 如果使用 LoRA
    "lora_alpha": 256,
}
```

## 5. 推理使用

```python
from transformers import LlavaForConditionalGeneration, AutoProcessor

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

# 多轮对话
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "这张图片里有什么？请详细描述。"},
        ],
    },
]

inputs = processor(text=conversation, images=[image], return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

## 6. LLaVA 版本演进

| 版本 | 关键改进 | VQA v2 | MMBench |
|------|---------|--------|---------|
| LLaVA-13B | 首个版本 | 71.3% | - |
| LLaVA-1.5-7B | MLP投影+更高分辨率 | 78.5% | 64.3% |
| LLaVA-1.5-13B | 更大 LLM | 80.0% | 67.7% |
| LLaVA-NeXT-72B | 动态切片+大模型 | 83.2% | 75.1% |

## 7. LLaVA 的贡献与影响

### 7.1 核心贡献

1. **简洁有效**：证明了简单的线性投影就足够桥接视觉和语言
2. **数据驱动**：GPT-4 辅助的指令数据构建成为标准做法
3. **开源社区**：开启了开源多模态助手的生态

### 7.2 后续工作

- **LLaVA-Med**：医学影像多模态助手
- **LLaVA-RLHF**：通过 RLHF 改善对齐
- **MiniGPT-4**：类似思路但使用 Q-Former
- **InternVL**：更大的视觉编码器

## 8. 小结

LLaVA 以极其简洁的设计实现了强大的多模态对话能力，核心在于：
视觉指令微调 + GPT-4 辅助数据构建 + 简单的线性投影。
LLaVA 证明了在多模态领域，**数据质量**比**模型复杂度**更重要，
这一发现深刻影响了后续的开源多模态模型设计。
