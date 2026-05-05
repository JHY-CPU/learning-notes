# 30_BART：去噪自编码器与生成任务

## 核心概念

- **BART (Bidirectional and Auto-Regressive Transformer)**：由 Facebook 于 2019 年提出，结合了 BERT 的双向编码器和 GPT 的自回归解码器。本质上是去噪自编码器。
- **去噪自编码器 (Denoising Autoencoder)**：BART 的预训练目标是对被噪声破坏的文本进行去噪恢复。模型需要学习理解（编码）和生成（解码）的联合能力。
- **噪声策略多样性**：BART 支持多种噪声注入方式——Token Masking、Token Deletion、Text Infilling（类似 T5 的 Span Corruption）、Sentence Permutation、Document Rotation。
- **Text Infilling**：BART 的主要噪声策略，与 T5 类似但使用单个 [MASK] 标记替换任意长度的连续 token 序列（长度由泊松分布决定）。
- **架构融合**：编码器使用双向注意力（全面理解输入），解码器使用因果注意力（自回归生成）。编码器-解码器之间通过交叉注意力连接。
- **与 T5 的比较**：BART 和 T5 都使用 Encoder-Decoder 架构，但噪声策略不同——BART 使用更多样化的噪声，且输出端需要生成完整的原始文本（而 T5 只生成被遮盖的片段）。
- **微调灵活性**：BART 可微调用于各种任务——分类（使用编码器输出）、生成（使用解码器输出）、理解（将编码器表示用于下游）。

## 数学推导

BART 的预训练：给定原始文本 $\mathbf{x}$，对其施加噪声得到 $\tilde{\mathbf{x}}$。

预训练目标——最大化给定噪声输入时原始文本的对数似然：
$$
\mathcal{L}_{\text{BART}} = -\log P(\mathbf{x} | \tilde{\mathbf{x}})
$$

通过 Encoder-Decoder 建模：
$$
P(\mathbf{x} | \tilde{\mathbf{x}}) = \prod_{t=1}^{T} P(x_t | x_{<t}, \text{Encoder}(\tilde{\mathbf{x}}))
$$

**噪声策略示例**（Text Infilling）：
- 原始：I love natural language processing
- 破坏：I love [MASK] language [MASK]
- 目标：I love natural language processing

## 直观理解

- **BART 像"修复破碎文本的专家"**：给你一篇被人用不同方式破坏的文章——有的词被涂黑（Token Masking），有的句子被颠倒了顺序（Sentence Permutation），甚至有些段被删了（Token Deletion）。你需要把它恢复成原文。这个过程让你既学会理解（找出哪些地方被改了），也学会生成（把改坏的地方修复）。
- **BART vs T5**：BART 需要恢复整篇原文（像全文听写），而 T5 只需补全被遮盖的片段（像填空）。BART 的学习任务更难，但也让 BART 在某些生成任务上表现更好。
- **多样化的噪声 = 更鲁棒的理解**：使用多种噪声策略训练，使 BART 对各种形式的文本破损（拼写错误、语序混乱）都有很强的容错能力。

## 代码示例

```python
from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# 加载 BART 模型
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')

# 1. 文本摘要
article = (
    "Natural language processing (NLP) is a subfield of linguistics, computer science, "
    "and artificial intelligence concerned with the interactions between computers and "
    "human language. The goal is to enable computers to understand, interpret, and "
    "generate human language in a way that is both meaningful and useful."
)
inputs = tokenizer(article, return_tensors='pt', max_length=1024, truncation=True)
with torch.no_grad():
    summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=4)
print("摘要:", tokenizer.decode(summary_ids[0], skip_special_tokens=True))

# 2. 分类任务（使用编码器输出）
from transformers import BartForSequenceClassification
cls_model = BartForSequenceClassification.from_pretrained(
    'facebook/bart-base', num_labels=2
)
inputs = tokenizer("This movie is great!", return_tensors='pt')
with torch.no_grad():
    outputs = cls_model(**inputs)
print(f"分类 logits: {outputs.logits}")

# 3. 条件生成（对话回复示例）
context = "Customer: I need help with my account.\\nSupport:"
inputs = tokenizer(context, return_tensors='pt')
with torch.no_grad():
    reply_ids = model.generate(inputs['input_ids'], max_length=30)
print("回复:", tokenizer.decode(reply_ids[0], skip_special_tokens=True))
```

## 深度学习关联

- **生成式预训练的标杆**：BART 是生成式预训练的代表模型之一，与 T5 共同展示了预训练 Encoder-Decoder 架构在生成任务上的潜力。为 BART 的后续工作（如 BART-large、mBART）提供了基础。
- **多语言扩展**：mBART 将 BART 扩展到多语言场景，使用多种语言的语料联合预训练，在无监督/低资源机器翻译等领域取得突破。
- **去噪预训练的启示**：BART 的"去噪"思想影响了后续的多模态预训练模型（如 BEiT、MaskGIT），它们在图像、音频等模态上类似地采用了"破坏-恢复"的预训练范式。
