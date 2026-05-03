# 28_GPT-3：In-context Learning 上下文学习

## 核心概念

- **GPT-3 (Generative Pre-trained Transformer 3)**：OpenAI 于 2020 年发布，拥有 1750 亿参数，是当时最大的稠密语言模型。展示了强大的 In-context Learning 能力。
- **In-context Learning (上下文学习)**：无需更新模型参数，仅通过输入中包含的示例（demonstrations）让模型学会执行新任务。模型从输入的上下文中"理解"任务格式和要求。
- **Few-shot / One-shot / Zero-shot**：
  - Few-shot：提供几个（通常 1-64 个）完整的输入-输出示例后让模型执行
  - One-shot：只提供一个示例
  - Zero-shot：仅用指令描述任务，无示例
- **提示工程 (Prompt Engineering)**：由于模型能力高度依赖于输入 prompt 的措辞、格式和示例选择，prompt 设计成为使用 GPT-3 的关键技能。
- **规模效应的量变到质变**：GPT-3 的 1750 亿参数相比 GPT-2 的 15 亿参数（约 100 倍）带来了质的变化——在算术、文章生成、代码生成等任务上展现出远超 GPT-2 的能力。
- **训练数据**：使用 CommonCrawl（经过过滤和去重）、WebText2、Books、Wikipedia 等总计约 570GB 的训练数据。
- **局限性**：推理能力不稳定（容易受 prompt 措辞影响）、缺乏真实理解（可能产生"合理但错误"的答案）、世界知识截止于训练数据收集时间、计算成本极高。

## 数学推导

In-context Learning 的形式化：
$$
\hat{y} = \arg\max_y P_{\text{GPT-3}}(y | \text{Prompt})
$$

其中 Prompt 包含：
$$
\text{Prompt} = \text{Instruction} + (x_1, y_1) + (x_2, y_2) + \ldots + (x_k, y_k) + x_{\text{query}}
$$

$(x_i, y_i)$ 是示例对，$x_{\text{query}}$ 是待预测输入。

**关键洞察**：GPT-3 参数 $\theta$ 在推理时完全固定。模型的所有"学习"都发生在前向传播的注意力层中——示例中的映射关系通过自注意力机制被"参照"到新的查询上。

## 直观理解

- **In-context Learning 像"照猫画虎"**：你给 GPT-3 看几个例子——"good -> 好, bad -> 差, happy -> ？"，它就会回答"快乐"。你没有改变它的任何知识，只是告诉它"现在做翻译"的格式。
- **不是真正的学习**：GPT-3 的"学习"是在推理时发生的——更准确地说，是在 example 和 query 之间建立注意力连接。这就像一个看了很多配对数据的统计学家，一看到"->"格式就知道要做什么。
- **Few-shot 的效果**：示例数量增加通常会提升表现，但存在边际递减。示例的选择也很重要——越相关、越代表性、顺序越合理的示例，效果越好。

## 代码示例

```python
# 由于 GPT-3 需要 API，这里用 HuggingFace 的相当模型演示概念
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 使用类似 GPT-3 的模型（这里用 opt-1.3b 演示 in-context learning）
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-1.3b')
model = AutoModelForCausalLM.from_pretrained('facebook/opt-1.3b')
tokenizer.pad_token = tokenizer.eos_token

# Few-shot 示例：情感分类
few_shot_prompt = """Review: This movie is great!
Sentiment: Positive

Review: I hated this film.
Sentiment: Negative

Review: The acting was wonderful.
Sentiment: Positive

Review: What a waste of time.
Sentiment:"""

inputs = tokenizer(few_shot_prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id
    )

result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("Few-shot Prompt 结果:")
print(result)

# Zero-shot 版本：无示例
zero_shot_prompt = "What is the sentiment of \"I love this\"? Sentiment:"
inputs = tokenizer(zero_shot_prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=5,
        pad_token_id=tokenizer.eos_token_id
    )
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"\nZero-shot 结果: {result}")
```

## 深度学习关联

- **LLM 范式的转折点**：GPT-3 标志着 NLP 从"预训练 + 微调"范式转向"预训练 + 提示"范式。不再需要为每个任务微调模型，使非研究人员也能使用 LLM。
- **大规模涌现能力的论证**：GPT-3 系统性地证明了模型规模对涌现能力的预测性——某些能力在模型达到一定规模前几乎为零，超过阈值后急剧提升。这一发现指导了后续更大模型（GPT-4、PaLM、LLaMA）的训练决策。
- **Prompt 工程的兴起**：GPT-3 的 In-context Learning 催生了 Prompt Engineering 和 Prompt Tuning 等新研究方向，包括 Chain-of-Thought、Few-shot 选择策略等技巧。
