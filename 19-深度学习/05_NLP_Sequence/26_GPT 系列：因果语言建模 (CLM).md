# 26_GPT 系列：因果语言建模 (CLM)

## 核心概念

- **因果语言建模 (Causal Language Modeling, CLM)**：自回归语言建模目标——基于之前所有 token 预测下一个 token。这是 GPT 系列使用的预训练范式，满足 $P(x) = \prod_{t=1}^{T} P(x_t | x_{<t})$。
- **因果掩码 (Causal Mask)**：在 Transformer 解码器中应用上三角掩码矩阵，使每个位置只能关注自己和之前的位置。确保信息流是单向的（从左到右）。
- **生成式预训练 (Generative Pre-Training)**：GPT 的核心思想——先在大规模语料上通过 CLM 预训练，再在具体任务上微调或直接进行零样本/少样本推理。
- **单向 vs 双向**：与 BERT 的双向 MLM 不同，GPT 只能从左到右看到过去信息。这使得 GPT 天生适合生成长文本，但在理解任务上弱于 BERT。
- **统一的文本生成框架**：所有 NLP 任务被统一为"文本生成"——将任务指令和输入拼接为 prompt，让 GPT 自回归生成答案。不需要为每个任务修改模型结构。
- **Decoder-only 架构**：GPT 只使用 Transformer Decoder（去掉交叉注意力），因为不需要编码器——所有任务都视为条件文本生成。
- **Scale Law**：GPT 系列证明了模型规模（参数量、数据量、计算量）越大，性能持续提升。这推动了大模型竞赛。

## 数学推导

因果语言建模的目标函数：
$$
\mathcal{L}_{\text{CLM}} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)
$$

其中条件概率通过 Transformer Decoder 建模：
$$
P(x_t | x_{<t}) = \text{softmax}\left(W_e h_t^{(\text{last})} + b\right)
$$

$h_t^{(\text{last})}$ 是最后一层解码器在位置 $t$ 的隐藏状态，$W_e$ 通常与输入嵌入矩阵共享权重（weight tying）。

对于自回归生成，解码过程为：
$$
x_{t} \sim P(x_t | x_{<t}; \theta)
$$

逐 token 生成整个序列。

## 直观理解

- **因果语言建模像填空式续写**：给你"今天天气真"，让你继续写"好"、"糟糕"、"热"等。每一步只基于已经写出的内容决定下一个词。就像写作文时一笔一笔地写，不能回头改。
- **单向 vs 双向的区别**：读句子"我___苹果"时，GPT 只能看到"我"来预测下一个词（可能猜"吃"、"喜欢"等），而 BERT 能看到"我___苹果"整句来猜空白。GPT 适合"创作"，BERT 适合"理解"。
- **Decoder-only 的优雅**：不需要编码器，不需要复杂的 Encoder-Decoder 架构，一个解码器就能处理所有任务——翻译（"将以下英文翻译成中文：Hello →"）、问答（"Q: 什么是 AI? A: "）、摘要（"总结：... →"）。一切皆是生成。

## 代码示例

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# 加载 GPT-2 模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# 因果语言建模的损失计算
text = "Artificial intelligence will transform"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs, labels=inputs['input_ids'])
    loss = outputs.loss
    logits = outputs.logits

print(f"CLM 损失（困惑度指标）: {loss.item():.4f}")
print(f"困惑度: {torch.exp(loss).item():.2f}")

# 自回归生成续写
input_ids = tokenizer.encode("The future of AI is", return_tensors='pt')
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id
    )
generated = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"\n生成结果:\n{generated}")
```

## 深度学习关联

- **GPT 系列的影响**：GPT 开创了"预训练 + 提示"（pretrain + prompt）范式，启发了 GPT-3 的 in-context learning、ChatGPT 的对话微调，以及整个大语言模型生态。
- **BERT vs GPT 的互补**：BERT 擅长 NLU 任务（分类、NER、QA），GPT 擅长 NLG 任务（写作、对话、翻译）。后续的 Encoder-Decoder 模型（T5、BART）试图统一两者，而 GPT 的 Decoder-only 路线最终在规模效应下胜出。
- **自回归生成的挑战**：自回归生成的核心挑战包括：暴露偏差（训练/推理差异）、生成速度慢（无法并行）、重复问题。这些挑战催生了多种解码策略（beam search、top-k/p 采样）和加速方法（投机解码、KV cache）。
