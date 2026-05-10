# SFT监督微调 - 大模型训练工程

*Full Fine-tuning / LoRA / Packing / Chat Template / 多轮对话*


# SFT 监督微调


Full Fine-tuning / LoRA / Packing / Chat Template / 多轮对话

[88-大模型训练工程](../index.html)
>
[03-训练实战](./)
>
            02_SFT监督微调

### 目录


1. [SFT 概述](#overview)
2. [Full Fine-tuning](#full-ft)
3. [LoRA 微调](#lora)
4. [Packing 策略](#packing)
5. [Chat Template](#chat-template)
6. [多轮对话数据构造](#multi-turn)
7. [训练 Epoch 选择](#epoch)
8. [最佳实践](#best-practice)


## 1. SFT 概述


监督微调（Supervised Fine-Tuning, SFT）是将预训练模型适配到特定任务或对话格式的关键步骤。通过在高质量的指令-回答对上训练，模型学会遵循指令并生成有用的回答。


### 1.1 训练范式对比


```
大模型训练三阶段:

阶段一: 预训练 (Pre-training)
┌─────────────────────────────────────────┐
│ 海量无标注文本 → 学习语言知识和世界知识    │
│ 数据: 1T+ tokens                         │
│ 方法: 自回归语言建模 (Next Token)         │
└─────────────────────────────────────────┘
                    │
                    ▼
阶段二: 监督微调 (SFT)
┌─────────────────────────────────────────┐
│ 指令-回答对 → 学习遵循指令               │
│ 数据: 10K-1M 条高质量样本                │
│ 方法: 指令跟随训练 (只在回答部分计算损失)  │
└─────────────────────────────────────────┘
                    │
                    ▼
阶段三: 对齐 (RLHF / DPO)
┌─────────────────────────────────────────┐
│ 人类偏好数据 → 学习人类偏好              │
│ 数据: 100K-1M 条偏好对                   │
│ 方法: PPO / DPO / KTO                   │
└─────────────────────────────────────────┘
```


### 1.2 Full FT vs LoRA 选择


| 特性 | Full Fine-tuning | LoRA |
| --- | --- | --- |
| 可训练参数 | 全部参数 | 0.1-1% 参数 |
| 显存需求 | 高 (与预训练相同) | 低 (仅需 LoRA 参数的梯度) |
| 训练速度 | 慢 | 快 |
| 效果上限 | 最高 | 接近 Full FT |
| 适用场景 | 大量数据 + 充足算力 | 数据有限 / 快速迭代 |


## 2. Full Fine-tuning


### 2.1 超参数设置


| 超参数 | 推荐值 | 说明 |
| --- | --- | --- |
| 学习率 | 1e-5 ~ 5e-5 | 远小于预训练学习率 |
| Batch Size | 64-256 (samples) | 根据 GPU 数量调整 |
| Epoch | 1-3 | 过多 epoch 容易过拟合 |
| Warmup | 3-10% 总步数 | 比预训练更短 |
| Weight Decay | 0.01 ~ 0.1 | 防止过拟合 |
| Max Length | 2048-4096 | 根据数据分布设定 |


### 2.2 损失计算策略


SFT 训练的关键是**只在回答部分计算损失**，不对 prompt 部分计算损失：


```
python
# SFT 损失计算 — 只在回答部分计算
def compute_sft_loss(logits, labels, prompt_length):
    # logits: [batch, seq_len, vocab_size]
# labels: [batch, seq_len]  (prompt部分为 -100)
# 将 prompt 部分的 label 设为 -100 (CrossEntropy 忽略)
    labels[:, :prompt_length] = -100
# 标准交叉熵损失
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100
    )
    return loss

# HuggingFace Trainer 中的实现
class SFTTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss  # HF 自动处理 -100 忽略
return (loss, outputs) if return_outputs else loss
```


## 3. LoRA 微调


LoRA（Low-Rank Adaptation）通过在冻结原模型参数的基础上添加低秩矩阵来实现高效微调。


### 3.1 基本用法


```
python
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                    # LoRA rank
    lora_alpha=32,           # alpha (通常 = 2 × rank)
    lora_dropout=0.05,       # dropout
    target_modules=[         # 目标模块
"q_proj", "k_proj", "v_proj", "o_proj",  # 注意力
"gate_proj", "up_proj", "down_proj" # FFN
    ],
    bias="none",
    modules_to_save=None,
)

# 应用 LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 41,943,040 || all params: 6,742,609,920 || trainable%: 0.62%
```


### 3.2 Target Modules 选择


| 方案 | 目标模块 | 参数量 | 效果 |
| --- | --- | --- | --- |
| 最小 | q_proj, v_proj | ~0.1% | 基础 |
| 推荐 | q_proj, k_proj, v_proj, o_proj | ~0.3% | 好 |
| 全面 | Attention + FFN 所有线性层 | ~0.6% | 最佳 |
| 嵌入层 | 所有线性层 + embed_tokens + lm_head | ~1% | 适配新词表 |


## 4. Packing 策略


Packing 将多条短样本拼接成一条长序列，减少 padding 浪费，显著提升训练效率。


### 4.1 问题：Padding 浪费


```
无 Packing (大量 padding 浪费):

Sample 1: [prompt_1][answer_1][PAD][PAD][PAD][PAD][PAD][PAD]
Sample 2: [prompt_2][answer_2][PAD][PAD]
Sample 3: [prompt_3][answer_3][PAD][PAD][PAD][PAD]

有效 token 比例: 很低 (短序列 padding 严重)

────────────────────────────────────
有 Packing (多条拼接):

[Prompt1][Answer1][EOS][Prompt2][Answer2][EOS][Prompt3][Answer3][EOS]

有效 token 比例: 接近 100%
```


### 4.2 实现


```
python
# HuggingFace SFTTrainer 内置 packing
from trl import SFTConfig, SFTTrainer

training_args = SFTConfig(
    output_dir="./output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    max_seq_length=2048,
    packing=True,             # 启用 packing
    dataset_text_field="text",
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# packing 会自动将短样本拼接
# 使用 attention_mask 区分不同样本的边界
```


> **Note:** **Packing 的挑战：**
> Packing 会将不同样本拼接在一起，需要注意 attention mask 的处理，确保模型不会 attend 到其他样本的 token。常用方法是使用
> `position_ids`
> 重置和自定义 attention mask。


## 5. Chat Template


Chat Template 定义了对话的格式化方式，确保模型能区分系统提示、用户输入和助手回答。


### 5.1 常见模板格式


```
text
# ChatML 格式 (GPT-4, Qwen)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
你好，请介绍一下自己。<|im_end|>
<|im_start|>assistant
你好！我是一个AI助手...<|im_end|>

# LLaMA 2 格式
<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

你好，请介绍一下自己。[/INST]
你好！我是一个AI助手... </s>

# LLaMA 3 格式
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

你好，请介绍一下自己。<|eot_id|><|start_header_id|>assistant<|end_header_id|>

你好！我是一个AI助手...<|eot_id|>
```


### 5.2 Jinja2 模板


```
python
# tokenizer_config.json 中的 chat_template
# ChatML Jinja2 模板
"""
{% for message in messages %}
{% if message['role'] == 'system' %}
<|im_start|>system
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'user' %}
<|im_start|>user
{{ message['content'] }}<|im_end|>
{% elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{ message['content'] }}<|im_end|>
{% endif %}
{% endfor %}
{% if add_generation_prompt %}
<|im_start|>assistant
{% endif %}
"""
# Python 中使用
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("model_name")

messages = [
    {"role": "system", "content": "你是一个有帮助的助手"},
    {"role": "user", "content": "什么是机器学习？"},
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```


## 6. 多轮对话数据构造


### 6.1 数据格式


```
json
[
    {
        "conversations": [
            {"role": "system", "content": "你是一个专业的数学老师。"},
            {"role": "user", "content": "什么是微积分？"},
            {"role": "assistant", "content": "微积分是数学的一个分支..."},
            {"role": "user", "content": "它的应用有哪些？"},
            {"role": "assistant", "content": "微积分广泛应用于..."}
        ]
    }
]
```


### 6.2 多轮损失掩码策略


```
多轮对话中的损失掩码:

Token:   [SYS] [U1] [A1] [U2] [A2]
Loss:    否    否   是   否   是

即: 只计算 assistant 回复部分的损失
    system 和 user 部分不计算损失

完整序列:
<|im_start|>system\n你是一个助手<|im_end|>\n
<|im_start|>user\n什么是ML？<|im_end|>\n
<|im_start|>assistant\nML是...<|im_end|>\n
<|im_start|>user\n应用？<|im_end|>\n
<|im_start|>assistant\n应用于...<|im_end|>

Label:   -100  -100 ... -100 [实际token] -100 -100 [实际token]
```


### 6.3 数据质量


> **Tip:** **高质量 SFT 数据的标准：**
>
> 1. 回答准确且完整
> 2. 指令清晰、无歧义
> 3. 格式一致（使用统一的模板）
> 4. 覆盖多种任务类型
> 5. 长度分布合理（避免全是短回答或长回答）
> 6. 去除重复和低质量样本


## 7. 训练 Epoch 选择


### 7.1 经验法则


| 数据量 | 推荐 Epoch | 说明 |
| --- | --- | --- |
| < 10K | 10-20 | 数据少，需要多次重复 |
| 10K - 100K | 3-5 | 适中 |
| 100K - 1M | 1-2 | 数据充足，避免过拟合 |
| > 1M | 1 | 一次即可，过多会过拟合 |


### 7.2 过拟合检测


- **训练 loss 持续下降，验证 loss 上升：**
   明确过拟合
- **模型过于重复某些回答：**
   过拟合训练数据
- **泛化能力下降：**
   在未见过的任务上表现变差


## 8. 最佳实践


> **Tip:** ### SFT 训练清单
>
>
> 1. 数据质量 > 数据数量（宁少勿滥）
> 2. 使用统一的 Chat Template
> 3. 只在 assistant 部分计算损失
> 4. 启用 Packing 提升效率
> 5. 学习率：1e-5 ~ 5e-5 (Full FT) 或 1e-4 ~ 3e-4 (LoRA)
> 6. Epoch：1-3（根据数据量调整）
> 7. 定期在验证集上评估
> 8. 保留最佳 checkpoint（基于验证 loss）

大模型训练工程 - SFT监督微调 | 最后更新: 2025年


<!-- Converted from: 02_SFT监督微调.html -->
