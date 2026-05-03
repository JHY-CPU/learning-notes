# 22_BERT：下一句预测 (NSP) 任务的争议

## 核心概念

- **NSP (Next Sentence Prediction)**：BERT 的第二个预训练任务。输入两个句子 A 和 B，预测 B 是否是 A 的下一句。50% 为正例（真实下一句），50% 为负例（随机抽取的句子）。
- **NSP 的设计目的**：提升需要理解句子间关系的下游任务性能，如问答 (QA)、自然语言推理 (NLI)。[CLS] 标记的最终表示用于二分类。
- **RoBERTa 的反驳**：Liu et al. (2019) 在 RoBERTa 中通过实验证明，移除 NSP 后模型性能反而提升或持平。这引发了关于 NSP 有效性的广泛讨论。
- **NSP 任务过于简单**：负例通常来自不同文档，模型只需要判断"话题是否一致"就能解决 NSP，而非真正学习句子间的逻辑关系。
- **SOP (Sentence Order Prediction) 替代**：ALBERT 提出的改进——不再使用不同文档的句子作为负例，而是将同一文档的连续两句交换顺序。模型必须判断"两句顺序是否正确"而非"是否来自同一文档"。
- **其他替代方案**：ELECTRA 使用替换 token 检测 (RTD) 替代 MLM+NSP；SpanBERT 使用跨句的 span 预测替代 NSP。

## 数学推导

NSP 的任务目标：
$$
\mathcal{L}_{\text{NSP}} = -\left[y \log P(\text{IsNext}) + (1-y) \log P(\text{NotNext})\right]
$$

其中 $y=1$ 表示 B 是 A 的下一句，$y=0$ 表示不是。$P(\text{IsNext})$ 基于 [CLS] 标记的表示通过二分类器得到。

BERT 的联合预训练损失：
$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

在 BERT 的原始实验中，移除 NSP 会影响 QNLI（问题蕴含识别）和 MRPC（释义检测）等任务的表现，但对其他任务的帮助不明显。

## 直观理解

- **NSP 像"句子拼图"**：给你两句话，"今天天气很好"和"我们去公园散步了"，问你第二句是否应该紧跟在第一句后面。听起来合理，但模型只需判断话题是否一致——两者都是关于天气和户外活动。
- **问题在于太简单**：如果负例来自完全不同的文章，模型只需要检测"话题是否突变"就行，根本不需要理解句子间的因果、时序等逻辑关系。这就像考学生"这两段话是否来自同一本书"——太容易了。
- **SOP 的改进**：ALBERT 把题目改成"今天我们去公园散步了"和"今天天气很好"，问顺序是否正确。模型需要理解"因为天气好，所以去散步"的逻辑关系，难度大得多。

## 代码示例

```python
from transformers import BertTokenizer, BertForNextSentencePrediction
import torch

# 加载 BERT+NSP 模型
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForNextSentencePrediction.from_pretrained('bert-base-chinese')

# 正例：两个连续的句子
sentence_a = "今天天气很好。"
sentence_b = "我们去公园散步了。"  # 合理的下一句

# 负例：两个不相关的句子
sentence_c = "机器学习是人工智能的一个分支。"

inputs = tokenizer(sentence_a, sentence_b, return_tensors='pt')
with torch.no_grad():
    outputs = model(**inputs)
    prob_positive = torch.softmax(outputs.logits, dim=1)

inputs_neg = tokenizer(sentence_a, sentence_c, return_tensors='pt')
with torch.no_grad():
    outputs_neg = model(**inputs_neg)
    prob_negative = torch.softmax(outputs_neg.logits, dim=1)

print(f"正例 (A→B) - IsNext: {prob_positive[0][0]:.4f}, NotNext: {prob_positive[0][1]:.4f}")
print(f"负例 (A→C) - IsNext: {prob_negative[0][0]:.4f}, NotNext: {prob_negative[0][1]:.4f}")
```

## 深度学习关联

- **预训练任务设计的重要性**：NSP 的争议凸显了预训练任务设计的重要性。一个好的预训练任务应该具有挑战性、需要深层理解而非浅层模式匹配。这直接影响了后续模型（ELECTRA、ALBERT、RoBERTa）的训练策略。
- **对比学习的早期探索**：NSP 本质上是对比学习的一种形式（区分正负句对），为后来的 SimCSE（句子表示的对比学习）等工作奠定了基础。
- **跨句理解任务的延续**：尽管 NSP 被质疑，理解句子间关系仍然是许多 NLP 任务（如推理、问答、对话）的核心能力。后续工作转向了更精细的跨句预训练目标。
