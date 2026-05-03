# 50_命名实体识别 (NER) 与 BIO 标注

## 核心概念

- **命名实体识别 (Named Entity Recognition, NER)**：从文本中识别出具有特定意义的实体（如人名、地名、机构名、时间、金额等），是信息抽取的基础任务之一。
- **BIO 标注体系 (Begin-Inside-Outside)**：最常用的序列标注格式。B-XXX 表示实体开始，I-XXX 表示实体内部，O 表示非实体。如"B-PER"表示"人名的开始"。
- **BIOES 标注体系**：BIO 的扩展版本，增加了 E（End，实体结束）和 S（Single，单个词构成实体），更精确地标注实体边界。
- **序列标注模型**：NER 通常建模为序列标注任务——对输入序列的每个 token 预测一个标签。经典模型是 Bi-LSTM + CRF，现代模型使用预训练语言模型（BERT、RoBERTa）直接输出标签。
- **CRF 层 (Conditional Random Field)**：在模型顶部添加 CRF 层捕捉标签之间的依赖关系（如 I-PER 不能直接跟在 B-LOC 后面）。CRF 层计算所有可能标签序列的概率，选择全局最优序列。
- **嵌套实体 (Nested Entities)**：实体可以嵌套，如"北京大学"中"北京"是地名，"北京大学"是机构名。BIO 标注难以处理嵌套，需要特殊处理（如使用多标签标注或 span-based 方法）。
- **评估指标**：严格评估要求实体边界和类型完全正确才算正确。使用精确率（Precision）、召回率（Recall）、F1 值评估。支持实体级别的微观评估。
- **常见数据集**：CoNLL-2003（英）、OntoNotes 5.0（多语种）、MSRA（中文）、ACE 2005（中英）。

## 数学推导

NER 作为序列标注问题：给定输入序列 $\mathbf{x} = (x_1, \ldots, x_n)$，预测标签序列 $\mathbf{y} = (y_1, \ldots, y_n)$，其中 $y_i \in \mathcal{L}$（标签集）。

**Bi-LSTM + CRF 模型**：
- Bi-LSTM 编码：$\mathbf{h}_i = [\overrightarrow{\text{LSTM}}(x_i); \overleftarrow{\text{LSTM}}(x_i)]$
- 发射分数：$E_{i, y} = W_{hy} \mathbf{h}_i + b_y$
- CRF 解码：$P(\mathbf{y} | \mathbf{x}) = \frac{\exp(\sum_i E_{i, y_i} + \sum_i T_{y_{i-1}, y_i})}{\sum_{\mathbf{y}'} \exp(\sum_i E_{i, y'_i} + \sum_i T_{y'_{i-1}, y'_i})}$

其中 $T_{y_{i-1}, y_i}$ 是转移分数。

**训练损失**（负对数似然）：
$$
\mathcal{L} = -\log P(\mathbf{y} | \mathbf{x})
$$

**Viterbi 解码**（推理时选择最优标签序列）：
$$
\mathbf{y}^* = \arg\max_{\mathbf{y}'} \sum_i E_{i, y'_i} + \sum_i T_{y'_{i-1}, y'_i}
$$

## 直观理解

- **BIO 标注像"词性标注的进阶版"**：小学时学过"名词 / 动词 / 形容词"的词性标注。NER 的 BIO 在此基础上更精细——不仅告诉你"这是一个专有名词"，还告诉你是"人名 (PER) / 地名 (LOC) / 机构名 (ORG)"。
- **CRF 层像"标签语法检查"**：没有 CRF 时，模型可能预测出"O B-LOC I-PER"这种不合理的序列。CRF 的作用就像"语法检查"，确保标签序列符合合理的模式——"B-XXX 后面必须是 I-XXX 或 O，不能直接跳到其他 B-XXX"。
- **嵌套实体的挑战**：句子"李先生在北京大学学习"："李"是 B-PER，但"北京大学"整体是 B-ORG，"北京"又是 B-LOC。这种"大套小"的关系就像俄罗斯套娃，BIO 的一个标签只能表示一个角色，无法同时表示"我既是 ORG 内部又是 LOC 的开始"。

## 代码示例

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

# 加载中文 NER 模型（BERT-base + CRF）
tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")

text = "李明在北京大学学习计算机科学。"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits.argmax(dim=-1)

# 解析标签
id2label = model.config.id2label
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
pred_labels = [id2label[p.item()] for p in predictions[0]]

print("NER 结果:")
for token, label in zip(tokens, pred_labels):
    if label != 'O':  # 只显示实体
        print(f"  {token}: {label}")

# 自定义 BIO 标注函数
def bio_encode(sentence, entities):
    """手动 BIO 标注示例"""
    # entities: [("李明", "PER"), ("北京大学", "ORG")]
    tags = ['O'] * len(sentence)
    for entity_text, entity_type in entities:
        start = sentence.find(entity_text)
        if start != -1:
            tags[start] = f'B-{entity_type}'
            for i in range(start + 1, start + len(entity_text)):
                tags[i] = f'I-{entity_type}'
    return list(zip(list(sentence), tags))

sentence = "李明在北京大学学习"
entities = [("李明", "PER"), ("北京大学", "ORG")]
result = bio_encode(sentence, entities)
print("\n手动 BIO 标注:")
for char, tag in result:
    print(f"  {char}: {tag}")
```

## 深度学习关联

- **预训练模型大幅提升 NER 效果**：BERT 等预训练模型将 NER 的 F1 值从 90%+ 提升到 95%+（CoNLL-2003），接近人类水平。预训练模型的双向上下文理解能力让模型更好地判断实体边界。
- **Span-based NER**：现代 NER 除了传统的序列标注方法，还有 span-based 方法——直接枚举所有可能的文本跨度并分类。这种方法天然支持嵌套实体。
- **生成式 NER**：随着 LLM 的发展，实体识别可以转化为"生成实体序列"而非"序列标注"。例如"提取文本中的人名和地名"，模型直接输出结构化的实体列表。这在 GPT-4 等模型中表现出色。
