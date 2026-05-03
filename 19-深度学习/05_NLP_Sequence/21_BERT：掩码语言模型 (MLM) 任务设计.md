# 21_BERT：掩码语言模型 (MLM) 任务设计

## 核心概念

- **BERT (Bidirectional Encoder Representations from Transformers)**：由 Google 于 2018 年提出，基于 Transformer Encoder 的预训练语言模型。核心创新是双向上下文建模。
- **掩码语言模型 (Masked Language Model, MLM)**：随机遮盖输入中 15% 的 token，让模型预测被遮盖的 token。这使得模型可以利用左右两侧的双向上下文信息。
- **MLM 的遮盖策略**：被选中的 15% token 中：80% 替换为 [MASK] 标记，10% 替换为随机词，10% 保持不变。这种策略迫使模型保持对上下文的依赖性，避免过度依赖 [MASK] 标记。
- **双向建模的优势**：相比 GPT 的从左到右因果建模，MLM 在每个位置都可以同时利用左右信息，学习到更丰富的上下文表示。
- **预训练-微调范式**：BERT 先在无标注语料上通过 MLM 进行预训练，再在下游任务上微调。一个预训练好的 BERT 可用于分类、NER、QA 等多个任务。
- **[CLS] 标记**：输入序列的第一个特殊标记，其最后一层的隐藏状态被用作整个序列的聚合表示，常用于分类任务。
- **MLM 的收敛速度**：由于每次只预测 15% 的 token，MLM 需要更多步数收敛，但每一步的计算效率比自回归生成更高。

## 数学推导

MLM 的预训练目标：
$$
\max_{\theta} \log P(\mathbf{x}_{\text{masked}} | \mathbf{x}_{\text{observed}})
$$

具体地，对第 $i$ 个被遮盖位置：
$$
P(x_i | \mathbf{x}_{\text{observed}}) = \frac{\exp(\mathbf{h}_i^\top e_{x_i})}{\sum_{v \in V} \exp(\mathbf{h}_i^\top e_v)}
$$

其中 $\mathbf{h}_i$ 是 BERT 最后一层在第 $i$ 位置的隐藏状态，$e_v$ 是词 $v$ 的嵌入向量。

总的 MLM 损失为所有被遮盖位置的交叉熵之和：
$$
\mathcal{L}_{\text{MLM}} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\text{observed}})
$$

其中 $\mathcal{M}$ 是被遮盖位置的集合。

## 直观理解

- **MLM 像高考完形填空**：文章中有 15% 的词被挖空，你需要根据前后文推算出每个空应该填什么。和高考不同的是，BERT 不仅看左边，还看右边——真正的双向理解。
- **遮盖策略的设计巧思**：80% 换 [MASK] 让模型学习真正预测缺失词；10% 换随机词迫使模型不过度依赖被遮盖位置的表示（因为可能被误导）；10% 不变让模型知道即使表面没变化也要检查上下文。这种"带噪声的预测"训练出了更强的鲁棒性。
- **[CLS] 标记像聚合报告**：在所有词的最前面放一个特殊标记 [CLS]，经过多层 Transformer 后，这个位置自然吸收了整个句子的信息——就像每个词都向 [CLS]"汇报"了自己的信息。

## 代码示例

```python
from transformers import BertTokenizer, BertForMaskedLM
import torch

# 加载预训练的 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertForMaskedLM.from_pretrained('bert-base-chinese')

# 输入句子，使用 [MASK] 遮盖需要预测的词
text = "今天天气真[MASK]，适合出去散步。"
inputs = tokenizer(text, return_tensors='pt')

# 预测
with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

# 获取 [MASK] 位置的预测结果
mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
mask_token_logits = predictions[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0]

print(f"输入: {text}")
print("预测 Top 5:")
for token_id in top_5_tokens:
    print(f"  {tokenizer.decode([token_id])}")

# 使用 BERT 做句子分类
from transformers import BertForSequenceClassification
cls_model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=2)
inputs = tokenizer("这部电影真好看", return_tensors='pt')
outputs = cls_model(**inputs)
print(f"\n分类 logits: {outputs.logits}")
```

## 深度学习关联

- **预训练范式的革命**：BERT 证明了 MLM 预训练 + 微调的有效性，开启了 NLP 的"预训练时代"。其影响力远超 NLP 本身，也影响了多模态模型（如 ViLT、BEiT）。
- **双向理解 vs 单向生成**：BERT 的双向理解能力使其在自然语言理解（NLU）任务上优于 GPT，但 GPT 的因果建模使其在自然语言生成（NLG）上更胜一筹。后续工作尝试融合两者（如 UniLM、GLM）。
- **RoBERTa 等的改进基础**：BERT 的 MLM 设计启发了大量改进工作——RoBERTa 优化了训练策略，ELECTRA 提出了替换 token 检测的更高效预训练任务。
