# 04_FastText：子词信息与 OOV 处理

## 核心概念

- **子词嵌入 (Subword Embedding)**：FastText 将每个词表示为其字符 n-gram 向量的和，而非直接学习整词的向量。例如 "apple" 在 n=3 时包含 "<ap", "app", "ppl", "ple", "le>" 等子词。
- **OOV (Out-of-Vocabulary) 处理**：由于子词 n-gram 可以组合出未见过的词，FastText 天然支持 OOV 词的向量表示——只需将 OOV 词的子词向量求和。
- **形态学信息利用**：子词表示能捕捉词缀、词根等形态信息，对形态丰富的语言（如德语、俄语、中文等）尤其有效。例如 "猫" 和 "猫咪" 共享部分子词。
- **与 Word2Vec 的关系**：FastText 本质上是在 Skip-gram 框架上的扩展，将输入词替换为子词 n-gram 集合，训练目标与负采样保持一致。
- **n-gram 哈希**：为避免子词数量爆炸，FastText 使用哈希函数将海量子词映射到固定大小的桶（bucket）中，控制模型大小。
- **词向量生成方式**：最终词向量为整词向量与所有子词 n-gram 向量的平均，或直接取子词的组合表示。

## 数学推导

FastText 的评分函数：
$$
s(w, c) = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g^\top \mathbf{v}_c
$$

其中 $\mathcal{G}_w$ 是词 $w$ 的所有子词 n-gram 集合（包括整词本身），$\mathbf{z}_g$ 是子词 $g$ 的向量表示，$\mathbf{v}_c$ 是上下文 $c$ 的向量。

负采样损失函数：
$$
\log\sigma(s(w, c)) + \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}[\log\sigma(-s(w_i, c))]
$$

对于 OOV 词 $w^*$，其向量表示为：
$$
\mathbf{v}_{w^*} = \frac{1}{|\mathcal{G}_{w^*}|}\sum_{g\in\mathcal{G}_{w^*}} \mathbf{z}_g
$$

## 直观理解

- **FastText 像中文偏旁部首**：看到不认识的字"氼"，你通过"氵"（水）和"入"推测它与水有关。FastText 也是如此，通过字符级子串理解未知词。
- **拼写错误容忍**：某人把"apple"打成"appel"，Word2Vec 完全无法处理，而 FastText 能通过共享子词"app"和"ple"给出合理嵌入。
- **形态学学习**：观察"-ing"后缀出现在"running"、"eating"、"walking"中，模型自然学到-ing 表示进行时。

## 代码示例

```python
from gensim.models import FastText

sentences = [
    ["猫", "喜欢", "追", "老鼠"],
    ["狗", "喜欢", "追", "猫"],
    ["猫咪", "是", "可爱", "的"]
]

# 训练 FastText 模型
model = FastText(
    sentences,
    vector_size=100,
    window=3,
    min_count=1,
    min_n=2,        # 最小 n-gram 长度
    max_n=4,        # 最大 n-gram 长度
    epochs=50
)

# 训练集内词
print("'猫'的向量:", model.wv["猫"][:5])

# OOV 词处理——模型从未见过"猫猫"这个整词
oov_vector = model.wv["猫猫"]
print("OOV词'猫猫'的向量:", oov_vector[:5])

# 相似词
print("类似'猫'的词:", model.wv.most_similar("猫", topn=3))
```

## 深度学习关联

- **子词 Tokenization 的启示**：FastText 的子词思想直接影响了 BPE (Byte-Pair Encoding) 和 WordPiece 等现代 tokenizer 的设计，后者已成为所有主流预训练模型的标配。
- **形态丰富的语言建模**：在 BERT 等模型中，子词嵌入使得模型对 morphologically rich languages 有了更好的处理能力，FastText 是这一趋势的开创者。
- **跨语言迁移的基础**：子词共享机制为跨语言模型（如 mBERT、XLM-R）提供了天然的基础——不同语言的词可能共享相同的字符或子词片段。
