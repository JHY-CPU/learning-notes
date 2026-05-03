# 38_指令微调 (Instruction Tuning) 原理

## 核心概念

- **指令微调 (Instruction Tuning)**：在预训练语言模型上，使用（指令，输出）格式的数据进行微调，使模型学会遵循人类指令完成任务。区别于普通微调，指令微调强调"学会理解任务描述"而非"记住具体任务"。
- **FLAN (Finetuned Language Models)**：Google 提出的指令微调范式，在大量 NLP 任务上收集指令格式数据，任务数量从几十到数千种不等。模型通过这种多任务训练学会"理解指令"的元能力。
- **任务格式化**：将不同任务统一为（指令，输入，输出）的模板格式。例如分类任务："判断以下句子的情感：{文本}" → "积极 / 消极"。翻译任务："将以下英文翻译为中文：{text}" → "{翻译}"。
- **多任务学习**：指令微调本质上是多任务学习——同时训练数十到数百个任务。不同任务共享模型参数，通过任务多样性增强泛化能力。
- **零样本泛化到新任务**：指令微调的核心目标是提升模型的零样本泛化能力——经过指令微调后，模型可以执行训练中未见过的任务（只要以指令形式描述）。
- **数据质量 vs 数量**：指令微调数据的质量（人工标注、指令清晰度、答案正确性）比数量更重要。FLAN 使用模板化构造，Self-Instruct 使用模型自生成，但人工质检仍然关键。
- **指令多样性**：指令的措辞多样性对泛化至关重要。同一个任务可以用多种方式描述（"翻译为"/"转换成英文"/"What is the English translation of"），避免模型过拟合到特定模板。

## 数学推导

指令微调的损失函数（与标准语言模型一致）：
$$
\mathcal{L}_{\text{IT}} = -\sum_{t} \log P(y_t | x_{<t}, \text{instruction}, \text{input})
$$

其中 $\text{instruction}$ 是任务描述，$\text{input}$ 是任务输入，$y_t$ 是期望输出的 token。

多任务联合训练：
$$
\mathcal{L}_{\text{total}} = \sum_{i=1}^{N} \lambda_i \cdot \mathcal{L}_i
$$

其中 $N$ 是任务数，$\lambda_i$ 是任务权重（通常按任务均匀采样或基于 token 数比例）。

## 直观理解

- **指令微调像"教导"而非"训练"**：预训练模型像一个博学但对人类指令格式不了解的学者。指令微调就像教他："当我说'翻译：...'时，请做翻译；当我说'总结：...'时，请做摘要。"这并不是教新知识，而是教"使用知识的方式"。
- **多任务就像上通识课**：与其只学数学（单一任务微调），不如同时学数学、物理、化学、文学（多任务指令微调）。不同学科之间的"学习如何学习"的能力可以迁移——学会了"回答问题的格式"，就等于学会了所有任务。
- **泛化到新任务像"触类旁通"**：模型没训练过"给广告语"，但它训练过"写诗"、"总结"、"创意写作"等任务。当你说"为这个产品写一句广告语"，它综合已有能力创造出新的输出——这就是指令微调的泛化能力。

## 代码示例

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 模拟指令微调的数据构建
instruction_data = [
    # 翻译任务
    {
        "instruction": "将以下英文翻译成中文",
        "input": "Machine learning is fascinating.",
        "output": "机器学习非常迷人。"
    },
    # 分类任务
    {
        "instruction": "判断以下句子的情感极性",
        "input": "这部电影太难看了。",
        "output": "消极"
    },
    # 摘要任务
    {
        "instruction": "用一句话总结以下文本",
        "input": "Transformer是一种基于自注意力机制的神经网络架构，广泛应用于自然语言处理领域。",
        "output": "Transformer是广泛使用的NLP架构。"
    },
]

# 格式化指令数据（FLAN 风格）
def format_instruction(example):
    return (
        f"指令: {example['instruction']}\n"
        f"输入: {example['input']}\n"
        f"输出: {example['output']}"
    )

for example in instruction_data:
    print("训练示例:")
    print(format_instruction(example))
    print()

# 使用预训练模型进行指令推理（zero-shot）
tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-small')
model = AutoModelForCausalLM.from_pretrained('google/flan-t5-small')

# 零样本测试指令泛化
test_instruction = "指令: 将这段话翻译成英文\n输入: 人工智能正在改变世界\n输出:"

inputs = tokenizer(test_instruction, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False
    )
print("Zero-shot 推理结果:")
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 深度学习关联

- **对齐人类意图的关键步骤**：指令微调使 LLM 从"下一个词预测器"转变为"指令遵循器"，是 ChatGPT 成功的关键技术之一。它使得模型输出更加有用、相关和安全。
- **从 InstructGPT 到 ChatGPT**：InstructGPT 使用指令微调 + RLHF 的范式，ChatGPT 继承了这一路线。指令微调提供了 RLHF 中初始模型的"基本指令遵循能力"，是后续对齐的重要前提。
- **Self-Instruct 的进化**：Self-Instruct 和 Alpaca 等展示了使用 GPT-3.5/4 自动生成指令数据来微调小模型的可能性，极大降低了指令微调的数据成本，推动了开源 LLM 生态（LLaMA、Vicuna、WizardLM 等）。
