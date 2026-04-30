# 67_多语言模型：mBERT 与 XLM-R

## 核心概念
- **多语言模型 (Multilingual Model)**：使用多种语言的语料联合预训练的模型，可以在一个模型中处理多种语言。核心能力是跨语言迁移——在资源丰富的语言上训练后，可以在资源匮乏的语言上表现良好。
- **mBERT (Multilingual BERT)**：Google 使用 Wikipedia 的 104 种语言联合预训练的 BERT 模型。共享 WordPiece 词汇表（约 110K token），所有语言共享相同的 Transformer 参数。
- **XLM-R (XLM-RoBERTa)**：Meta 使用 CommonCrawl 数据中的 100 种语言，基于 RoBERTa 训练策略的大规模多语言模型。使用更大的词汇表（250K）和更多的训练数据（2TB 过滤后的 CommonCrawl）。
- **跨语言迁移的机制**：共享嵌入空间意味着"苹果"在中文、"apple"在英文、"manzana"在西班牙文都被映射到附近的嵌入向量。Transformer 参数共享使模型学习到语言无关的语义表示。
- **语言混淆 (Language Confusion)**：多语言模型可能混淆语言的特定特征——如把西班牙语词的性别规则错误地应用到德语上。在低资源语言上更明显。
- **零样本跨语言迁移 (Zero-shot Cross-lingual Transfer)**：在英文数据上微调多语言模型，可以直接在中文、阿拉伯语等其他语言上使用——无需任何目标语言的标注数据。这是多语言模型的最大价值。
- **语言编码**：部分多语言模型（如 XLM 原生版）使用语言编码（language embedding）告诉模型当前输入是什么语言，但 mBERT 和 XLM-R 不显式使用语言编码——它们通过词汇表和训练数据自动学习语言信息。
- **性能对比**：XLM-R 通常在大多数低资源语言上优于 mBERT，但在某些高资源语言上可能与之持平。更大的词汇表和更多的训练数据是 XLM-R 优势的关键。

## 数学推导
**多语言联合预训练**（mBERT 的 MLM）：
给定 $L$ 种语言，每种语言的语料 $\mathcal{D}_l$：
$$
\mathcal{L} = \sum_{l=1}^{L} \sum_{\mathbf{x} \in \mathcal{D}_l} \mathbb{E}_{\text{mask}} \left[ -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\text{masked}}) \right]
$$

所有语言的模型参数 $\theta$ 完全共享。

**XLM 的翻译语言模型 (TLM)**：
XLM 原始版本中使用了 TLM——将句子对（源语言 + 目标语言）拼接，遮盖后预测。这使得模型可以跨语言对齐表示：
$$
\mathcal{L}_{\text{TLM}} = -\log P(x_{\text{masked}}^s, x_{\text{masked}}^t | \text{[src+sent], [tgt+sent]})
$$

**跨语言嵌入的相似性**：两种语言 $a$ 和 $b$ 的词嵌入分布可以通过"对齐程度"衡量：
$$
\text{Alignment} = \mathbb{E}_{x \in V_a, y \in V_b} [\cos(E_a(x), E_b(y))]
$$

好的多语言模型会使不同语言中语义相似的词向量更加接近。

## 直观理解
- **多语言模型像"全球百科全书编辑"**：一个编辑团队同时处理 100 种语言的 Wikipedia。团队共享同一个"大脑"（共享参数），虽然看不同语言的词（不同的 tokenizer），但理解和表达的是相同的知识。
- **跨语言迁移像"学霸帮你翻译"**：一个学霸（在英语上微调过的多语言模型）没有学过法语，但当给他法语问题时，他能利用"学好英语的方法"（共享的 Transformer 参数）来理解法语。这对低资源语言特别有价值——某些语言在互联网上的数据很少，但可以通过共享表示从高资源语言中受益。
- **零样本迁移**：就像你学会了下国际象棋，然后第一次看到中国象棋——虽然规则不同（语言不同），但棋子的走法和策略（底层语言能力）可以迁移。你不需要专门学习中国象棋就能开始玩。
- **XLM-R 的优势来源**：XLM-R 相比 mBERT，就像让同一个编辑团队读了更多书（2TB vs Wikipedia），并且用了更全的词典（250K vs 110K 词汇表）。

## 代码示例
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch

# 加载 mBERT 和 XLM-R
mbert_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
xlmr_tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

# 观察不同语言的分词结果
texts = {
    "English": "I love natural language processing.",
    "Chinese": "我喜欢自然语言处理。",
    "Arabic": "أنا أحب معالجة اللغة الطبيعية.",
    "Hindi": "मुझे प्राकृतिक भाषा प्रसंस्करण पसंद है।",
    "Korean": "저는 자연어 처리를 좋아합니다.",
}

print("不同语言的分词结果 (mBERT):")
for lang, text in texts.items():
    tokens = mbert_tokenizer.tokenize(text)
    token_count = len(tokens)
    # 词汇表中有多少 token 是完整的词
    full_words = sum(1 for t in tokens if not t.startswith("#"))
    print(f"  {lang:8}: {token_count:3d} tokens, {full_words} 完整词")

print(f"\nmBERT 词汇表大小: {mbert_tokenizer.vocab_size}")
print(f"XLM-R 词汇表大小: {xlmr_tokenizer.vocab_size}")

# 零样本跨语言情感分类演示
print("\n零样本跨语言情感分类:")
print("在英文数据上微调 → 直接应用于多种语言")

# 加载 XLM-R 情感分类模型
sentiment_model = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment"
)

texts_to_test = [
    ("English", "I love this product, it's amazing!"),
    ("Chinese", "这个产品太棒了，我非常喜欢！"),
    ("French", "Ce produit est fantastique !"),
    ("Spanish", "¡Este producto es increíble!"),
]

for lang, text in texts_to_test:
    result = sentiment_model(text)[0]
    print(f"  [{lang:8}] {text[:30]:30s} -> {result['label']} ({result['score']:.3f})")
```

## 深度学习关联
- **低资源语言 NLP 的关键**：多语言模型是解决低资源语言 NLP 问题最成功的方法之一。对于世界上约 7000 种语言中绝大多数"低资源"语言，多语言模型提供了从高资源语言迁移知识的能力。
- **跨语言多模态模型的基石**：多语言模型的思想扩展到多模态领域——如 CLIP (多语言版本) 和 M3AE (多语言多模态自编码器)。它们共享"跨语言、跨模态"的表示空间。
- **多语言模型的挑战**：多语言模型面临的挑战包括"公平性"（高资源语言表现好于低资源语言）、"语言混淆"、以及"词汇表分配"（需要平衡各语言的子词覆盖）。最新的 Llama 3 等模型虽然主要是英文，但也通过多语言数据扩展了语言覆盖。
