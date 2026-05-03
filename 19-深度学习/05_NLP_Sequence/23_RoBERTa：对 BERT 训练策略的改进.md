# 23_RoBERTa：对 BERT 训练策略的改进

## 核心概念

- **RoBERTa (Robustly Optimized BERT Approach)**：Meta 于 2019 年提出，通过系统的实验分析发现 BERT 的训练不足，在相同的架构下通过改进训练策略取得显著提升。
- **动态掩码 (Dynamic Masking)**：BERT 在数据预处理时一次性生成静态掩码（每个样本候选 10 次），而 RoBERTa 在每次输入数据时动态生成掩码模式。同一句子在不同 epoch 中被遮盖不同位置，增加了训练数据的多样性。
- **移除 NSP 任务**：RoBERTa 通过实验验证了移除 NSP 任务对大多数下游任务没有负面影响甚至有所提升，简化了预训练目标。
- **更大批次训练 (Large-Batch Training)**：使用更大的 batch size（8K）和更高的学习率训练，配合 Adam 优化器的线性预热。大 batch 使梯度更稳定，加速收敛。
- **更多数据 (More Data)**：使用 160GB 训练数据（原始 BERT 的 10 倍以上），包括 BookCorpus、CC-News、OpenWebText、Stories 等，证明数据量对预训练效果的关键作用。
- **更长训练步数**：训练步数从 BERT 的 100 万步增加到 500 万步，模型有更充分的收敛。
- **文本编码方式**：使用 BPE (Byte-Pair Encoding) 而非 BERT 的 WordPiece，基于字节级别的分词更通用。

## 数学推导

RoBERTa 的训练目标（简化后的 MLM 任务）：
$$
\mathcal{L} = -\sum_{i \in \mathcal{M}} \log P(x_i | \mathbf{x}_{\text{observed}})
$$

不包含 NSP 损失（对比 BERT 的 $\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}$）。

优化器使用 AdamW（带权重衰减的 Adam）：
$$
\theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda\theta_t \right)
$$

其中 $\lambda$ 是权重衰减系数，RoBERTa 设置 $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=1e-6$。

训练超参数对比：
- BERT：lr=1e-4, batch=256, steps=1M, warmup=10K
- RoBERTa：lr=6e-4 (peak), batch=8K, steps=500K, warmup=24K

## 直观理解

- **RoBERTa 像"把 BERT 重新好好训练一遍"**：BERT 当初发布时训练得不够充分——就像学生只做了 10 套题就考试了，而 RoBERTa 做了 100 套题（更多数据、更久训练），成绩自然更好。
- **动态掩码的好处**：静态掩码就像每次做同一道完形填空——掩掉的位置永远是固定的"今天天气真_MASK_"，做过一次就知道答案。动态掩码让同一个句子每次都被不同地方挖空，模型必须真正理解全文。
- **移除 NSP 的发现**：RoBERTa 发现 NSP 不仅没帮助，还可能有害——因为 MLM 本身已经蕴含了跨句信息（长上下文可达 512 token 已包含多个句子），额外增加 NSP 任务反而分散了模型的学习能力。

## 代码示例

```python
from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

# 加载 RoBERTa 模型和分词器
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')

# RoBERTa 使用 <mask> 标记
text = "I love watching <mask> movies."
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits

mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
mask_token_logits = predictions[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0]

print(f"输入: {text}")
print("预测 Top 5:")
for token_id in top_5_tokens:
    print(f"  {tokenizer.decode([token_id])}")

# 使用 RoBERTa 做分类
from transformers import RobertaForSequenceClassification
cls_model = RobertaForSequenceClassification.from_pretrained(
    'roberta-base', num_labels=2
)
inputs = tokenizer("This movie is great!", return_tensors='pt')
outputs = cls_model(**inputs)
print(f"\n分类结果: {outputs.logits}")
```

## 深度学习关联

- **训练策略 > 架构创新**：RoBERTa 的重要启示是——在相同架构下，优化训练策略（数据量、batch size、掩码策略）可以带来显著提升。这影响了后续所有预训练模型的训练流程设计。
- **大 batch 训练的验证**：RoBERTa 对大 batch 训练的成功实践，为后续大规模模型训练提供了重要参考。GPT-3、LLaMA 等模型都使用了大 batch 加梯度累积。
- **去芜存菁的预训练目标**：RoBERTa 对 NSP 的否定推动社区重新思考预训练任务设计，后续的 ELECTRA、SpanBERT 等提出了更高效的目标函数。
