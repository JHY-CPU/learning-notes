# 36_Tokenizer 技术：BPE, WordPiece, Unigram

## 核心概念

- **Tokenizer (分词器)**：将原始文本转换为模型可处理的 token 序列的组件。它是 NLP 模型的第一步，其质量直接影响模型性能。
- **BPE (Byte-Pair Encoding)**：由 Sennrich et al. (2016) 引入 NLP。从字符级开始，迭代地合并最频繁的相邻字符对，直到达到目标词汇表大小。GPT 系列、RoBERTa 使用 BPE。
- **WordPiece**：Google 提出的算法（由 Schuster & Nakajima 2012 引入）。与 BPE 类似，但合并标准基于"最佳化训练数据的似然值"而非频率。BERT、DistilBERT 使用 WordPiece。
- **Unigram Language Model**：Kudo (2018) 提出。从较大的词种子集开始，逐步移除使似然损失最小的 token。对每个子词序列，计算其概率为各 token 概率的乘积。
- **Byte-level BPE**：在字节级别（而非字符级别）运行 BPE，确保可以编码任意 Unicode 字符和特殊符号。GPT-2 首次使用，解决了 OOV 问题。
- **添加特殊 Token**：[CLS]、[SEP]、[MASK]、[PAD]、<s>、</s> 等特殊标记用于特定 NLP 任务。
- **预分词 (Pre-tokenization)**：在应用 BPE/WordPiece 之前，先用规则（如空格、标点）对文本进行初步切分。不同的预分词策略会影响最终 tokenizer 的行为。

## 数学推导

**BPE 算法**：
- 初始化词汇表为所有字符
- 重复直到词汇表大小达到目标：
   - 统计所有相邻 token 对的频率
   - 合并最频繁的一对
   - 将新合并的 token 加入词汇表

**WordPiece 合并标准**：
$$
\text{Score}(a, b) = \frac{\text{freq}(ab)}{\text{freq}(a) \cdot \text{freq}(b)}
$$

其中 $ab$ 是 token $a$ 和 $b$ 合并后的新 token。选择得分最高的对进行合并。

**Unigram 模型**：给定 token 序列 $\mathbf{x} = [x_1, \ldots, x_M]$，其概率为：
$$
P(\mathbf{x}) = \prod_{i=1}^{M} p(x_i)
$$

其中 $p(x_i)$ 是每个 token 的概率。从大词汇表开始，移除最小化似然损失的 token：
$$
\mathcal{L} = -\sum_{\text{sentences}} \log P(\text{sentence})
$$

## 直观理解

- **BPE 像"从笔画到单词"的逐步构建**：先从最基础的笔画（字符）开始，发现"的"和"确"经常一起出现 → 合并为"的确" → 发现"的确"和"实"常出现 → 合并为"实际上"。BPE 不关心语义，只关心统计共现频率。
- **WordPiece 像"语义驱动"的合并**：与 BPE 不同，WordPiece 合并时考虑"合并后带来的信息量"。如果"un"和"happy"各自很常见，但"unhappy"的出现频率远超它们随机共现的期望值，说明合并有信息量。
- **Unigram 像"减法"而非"加法"**：BPE/WordPiece 从下往上构建（从小到大的加法），Unigram 先有一个巨大的词汇表（所有可能的子词），然后逐步删除最不重要的 token，就像雕塑家从石头中"减法"出作品。
- **字节级 vs 字符级**：字节级 tokenizer 看到的是"字节"（如 UTF-8 字节值），而不是"字符"。这使得它可以处理任何语言的任何字符（包括 emoji），而不会有 OOV 问题。

## 代码示例

```python
from transformers import AutoTokenizer

# BPE Tokenizer（GPT-2）
gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
text = "Hello, I am learning tokenization!"
tokens = gpt2_tokenizer.tokenize(text)
ids = gpt2_tokenizer.encode(text)
print(f"GPT-2 (BPE) tokens: {tokens}")
print(f"GPT-2 ids: {ids}")
print(f"解码回去: {gpt2_tokenizer.decode(ids)}")

# WordPiece Tokenizer（BERT）
bert_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
tokens_bert = bert_tokenizer.tokenize(text)
ids_bert = bert_tokenizer.encode(text)
print(f"\nBERT (WordPiece) tokens: {tokens_bert}")
print(f"BERT ids: {ids_bert}")

# Unigram Tokenizer（XLNet）
xlnet_tokenizer = AutoTokenizer.from_pretrained('xlnet-base-cased')
tokens_xlnet = xlnet_tokenizer.tokenize(text)
ids_xlnet = xlnet_tokenizer.encode(text)
print(f"\nXLNet (Unigram) tokens: {tokens_xlnet}")

# 观察不同 tokenizer 处理同一中文文本
chinese_text = "今天天气真好，适合学习自然语言处理。"
for name, tok in [("GPT-2", gpt2_tokenizer), ("BERT", bert_tokenizer)]:
    print(f"\n{name} 中文分词: {tok.tokenize(chinese_text)}")
```

## 深度学习关联

- **Embedding 层的大小控制**：词汇表大小直接决定 Embedding 层参数量（$V \times d$）。BPE 通过控制合并次数精确控制 $V$，WordPiece 和 Unigram 也类似。一个合理的大小通常是 32K-256K。
- **子词共享的跨语言意义**：BPE/WordPiece 的子词为跨语言模型提供了天然共享单元——mBERT 在 104 种语言上共享同一个 WordPiece 词汇表，不同语言通过共享子词获得跨语言迁移能力。
- **Tokenizer 设计的前沿**：近年出现了一些改进方案——SentencePiece（将 BPE/Unigram 应用于原始文本，无需预分词，支持日语/中文）、MegaByte（千字节级 token）、以及 tokenizer-free 模型（如 ByT5、CANINE）。
