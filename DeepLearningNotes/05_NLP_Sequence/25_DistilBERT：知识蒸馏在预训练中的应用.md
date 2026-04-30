# 25_DistilBERT：知识蒸馏在预训练中的应用

## 核心概念
- **知识蒸馏 (Knowledge Distillation)**：用一个大模型（教师模型）的输出来训练一个小模型（学生模型），使学生模型模仿教师的行为。DistilBERT 首次将蒸馏应用于预训练阶段。
- **三合一损失函数**：DistilBERT 的蒸馏损失包含三个部分：蒸馏损失（匹配教师 softmax 输出）、MLM 损失（匹配原始标签）、余弦嵌入损失（对齐隐藏状态）。
- **教师-学生架构**：教师是 BERT-base（12 层），学生是 DistilBERT（6 层），学生层数减半。学生每 2 层对应教师的 1 层，通过动态匹配学习。
- **训练效率提升**：DistilBERT 的训练时间约为 BERT 的 10%，推理速度提升 60%，模型体积缩小 40%，同时保留了 BERT 约 97% 的语言理解能力。
- **无 NSP 任务**：DistilBERT 不使用 NSP 任务，继承了 RoBERTa 的发现。仅使用 MLM 进行蒸馏训练。
- **初始化策略**：学生模型从教师模型中每隔一层复制权重进行初始化，而非随机初始化。这显著加速了蒸馏收敛。
- **学生训练的软标签**：学生不仅学习硬标签（原始词预测），还学习教师的软标签概率分布（temperature scaling 后的 softmax 输出），后者包含了更丰富的类间关系信息。

## 数学推导
DistilBERT 的蒸馏损失函数：
$$
\mathcal{L} = \alpha \cdot \mathcal{L}_{\text{ce}} + \beta \cdot \mathcal{L}_{\text{dis}} + \gamma \cdot \mathcal{L}_{\text{cos}}
$$

**蒸馏损失**（使用 soft target）：
$$
\mathcal{L}_{\text{dis}} = -\sum_i p_i^T \log(p_i^S), \quad p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

其中 $T$ 是温度参数（$T > 1$ 使分布更平滑），$z_i$ 是 logits，上标 $T$, $S$ 分别表示教师和学生。

**MLM 损失**（使用 hard target 的真实标签）：
$$
\mathcal{L}_{\text{ce}} = -\sum_{i \in \mathcal{M}} \log \frac{\exp(z_i^S)}{\sum_j \exp(z_j^S)}
$$

**余弦嵌入损失**（对齐师生隐藏状态）：
$$
\mathcal{L}_{\text{cos}} = -\frac{1}{N} \sum_{i=1}^{N} \frac{\mathbf{h}_i^T \cdot \mathbf{h}_i^S}{\|\mathbf{h}_i^T\| \cdot \|\mathbf{h}_i^S\|}
$$

## 直观理解
- **知识蒸馏像"师带徒"**：教师 BERT 是一个经验丰富的老教授（12 层），学生 DistilBERT 是一个聪明的年轻人（6 层）。老教授不仅告诉学生正确答案，还解释"为什么这么觉得"（软标签概率分布），学生学得更快更透彻。
- **软标签的价值**：对于"苹果"的预测，硬标签只告诉学生"答案是苹果"，但软标签说"我认为 80% 是苹果，15% 是水果，5% 是梨"——学生从这种"近似但不完全正确"的答案中学到了类别之间的相似性关系。
- **温度参数的意义**：高温（T > 1）像把老师的"意见"调得更温和——原本非常确定的判断变得不那么极端，让学生看到更多细微的区分信息。低温（T = 1）就是原始判断。

## 代码示例
```python
from transformers import DistilBertTokenizer, DistilBertForMaskedLM
import torch
import time

# 加载 DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForMaskedLM.from_pretrained('distilbert-base-uncased')

text = "I love watching [MASK] movies."
inputs = tokenizer(text, return_tensors='pt')

# 推理速度测试
start = time.time()
with torch.no_grad():
    outputs = model(**inputs)
infer_time = (time.time() - start) * 1000

# 对比参数量
distilbert_params = sum(p.numel() for p in model.parameters())

# BERT-base 的参考参数量
bert_base_params = 110_000_000  # 约 1.1 亿
print(f"DistilBERT 参数量: {distilbert_params:,}")
print(f"BERT-base 参数量（参考）: {bert_base_params:,}")
print(f"参数压缩比: {distilbert_params / bert_base_params:.2%}")

# 预测结果
mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
mask_logits = outputs.logits[0, mask_token_index, :]
top_5 = torch.topk(mask_logits, 5, dim=1).indices[0]

print(f"\n输入: {text}")
print("预测 Top 5:")
for token_id in top_5:
    print(f"  {tokenizer.decode([token_id])}")
```

## 深度学习关联
- **预训练蒸馏的范式开创**：DistilBERT 开创了"先预训练大模型，再蒸馏为小模型"的范式，后续有 TinyBERT（引入任务特定蒸馏）、MiniLM（深层蒸馏）、MobileBERT（瓶颈架构蒸馏）等。
- **蒸馏在 LLM 中的应用**：知识蒸馏在大型语言模型中广泛应用——Alpaca、Vicuna 等通过蒸馏 GPT-3.5/4 的输出训练小模型，Phi-1/2 通过"教科书质量"数据的蒸馏训练高效代码模型。
- **模型压缩三剑客**：蒸馏 + 量化 + 剪枝构成了现代模型压缩的三大技术。DistilBERT 展示了蒸馏的威力，为部署场景中的模型优化提供了实用方案。
