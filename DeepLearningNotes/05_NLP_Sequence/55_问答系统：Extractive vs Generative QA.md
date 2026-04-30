# 55_问答系统：Extractive vs Generative QA

## 核心概念
- **问答系统 (Question Answering, QA)**：自动回答自然语言提问的系统。根据回答生成方式分为抽取式（Extractive）和生成式（Generative）两种。
- **抽取式问答 (Extractive QA)**：从给定的上下文中直接提取一个连续文本片段作为答案。典型任务如 SQuAD (Stanford Question Answering Dataset)——模型需要预测答案在原文中的起始和结束位置。
- **生成式问答 (Generative QA)**：理解上下文后生成自由形式的答案，不限定于原文中的连续片段。可以综合多个句子信息或用自己的语言组织答案。
- **Span 预测**：抽取式 QA 的核心任务——对于上下文中的每个位置，预测它是否是答案的起始/结束位置。使用两个分类头分别预测 start logits 和 end logits。
- **开放域问答 (Open-domain QA)**：无给定上下文，模型需要从大规模知识库或网络信息中检索并回答。通常采用"检索 + 阅读"（Retriever-Reader）或"检索 + 生成"（RAG）模式。
- **闭卷问答 (Closed-book QA)**：模型仅依靠自身参数中存储的知识（训练数据中学到的）回答问题。GPT-3 等大模型展示出强大的闭卷问答能力。
- **SQuAD 数据集**：基于 Wikipedia 文章的抽取式 QA 数据集，包含 10 万个（问题，上下文，答案）三元组。是抽取式 QA 的经典基准。
- **评估指标**：精确匹配 (Exact Match, EM) 和 F1 值（基于 token 重叠度）。F1 衡量预测和参考答案的 n-gram 重合度。

## 数学推导
**抽取式 QA（BERT 用于 SQuAD）**：
$$
\text{StartLogits}_i = W_{\text{start}}^\top h_i + b_{\text{start}}
$$

$$
\text{EndLogits}_i = W_{\text{end}}^\top h_i + b_{\text{end}}
$$

其中 $h_i$ 是 BERT 在第 $i$ 位置的隐藏状态。

答案 $a = \text{context}[s:e]$，选择最大化 $P_{\text{start}}(s) \cdot P_{\text{end}}(e)$ 的区间 $(s, e)$：
$$
P_{\text{start}}(s) = \frac{\exp(\text{StartLogits}_s)}{\sum_j \exp(\text{StartLogits}_j)}
$$

$$
P_{\text{end}}(e) = \frac{\exp(\text{EndLogits}_e)}{\sum_j \exp(\text{EndLogits}_j)}
$$

训练使用交叉熵损失：
$$
\mathcal{L} = -\log P_{\text{start}}(s^*) - \log P_{\text{end}}(e^*)
$$

其中 $(s^*, e^*)$ 是真实答案的起止位置。

## 直观理解
- **抽取式像"在书中划线"**：给你一本书（上下文）和一个问题，你找到答案所在的句子，然后划出答案对应的连续文本。答案必须是书中原文的一部分。
- **生成式像"用自己的话回答"**：同样给你一本书，读完后用自己的语言组织答案。可以总结、合并信息、解释概念——不限于原文词汇。
- **开放域问答像"开卷考试"**：没有指定的"上下文"，学生自己从整个图书馆（整个网络）找信息来回答问题。这比 SQuAD 难多了——先是信息检索，然后是阅读理解。
- **闭卷问答像"闭卷考试"**：一切靠"记忆"。GPT-3 的巨大参数量意味着它可以"记住"大量训练数据中的事实性知识（Paris 是法国的首都）。但闭卷就意味着知识是静态的——训练数据截止日期后的新信息就无法回答。

## 代码示例
```python
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import torch

# 1. 抽取式问答
qa_pipeline = pipeline("question-answer", model="deepset/bert-base-cased-squad2")

context = """
The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France.
It was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.
It is named after the engineer Gustave Eiffel, whose company designed and built the tower.
"""

question = "When was the Eiffel Tower built?"
result = qa_pipeline(question=question, context=context)
print(f"抽取式 QA:")
print(f"  问题: {question}")
print(f"  答案: {result['answer']}")
print(f"  置信度: {result['score']:.4f}")

# 2. 生成式问答（使用 T5）
t5_qa = pipeline("text2text-generation", model="google/flan-t5-large")
prompt = f"Answer the question based on the context.\nContext: {context}\nQuestion: {question}\nAnswer:"
result_t5 = t5_qa(prompt, max_length=30)
print(f"\n生成式 QA (Flan-T5):")
print(f"  答案: {result_t5[0]['generated_text']}")

# 3. 手动实现抽取式 QA 的 start/end 预测
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

inputs = tokenizer(question, context, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

start_scores = outputs.start_logits
end_scores = outputs.end_logits

# 取最高分的起止位置
start_idx = torch.argmax(start_scores)
end_idx = torch.argmax(end_scores)

# 解码答案
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
answer = tokenizer.convert_tokens_to_string(tokens[start_idx:end_idx+1])
print(f"\n手动预测答案: {answer}")
```

## 深度学习关联
- **预训练模型的 QA 能力**：BERT 在 SQuAD 上的表现（F1: 93.2%）是预训练模型能力的标志性证明。此后几乎所有 QA 系统都基于预训练模型。
- **长文本 QA 挑战**：SQuAD 的上下文通常只有 1 个段落（约 500 token），但现实场景中需要阅读多页文档。Longformer、BigBird、Reformer 等长文本模型的出现在一定程度上解决了这一问题。
- **从抽取式到生成式的范式转变**：大语言模型时代，生成式 QA（如 ChatGPT 的回答）日益流行。抽取式 QA 的优势（可验证、忠实于原文）在需要事实准确性的场景（法律、医疗）中仍然重要。
