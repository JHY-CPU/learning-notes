# 29_T5：文本到文本 (Text-to-Text) 统一框架

## 核心概念

- **Text-to-Text 框架**：T5 (Text-to-Text Transfer Transformer) 将所有 NLP 任务统一为"输入文本 → 输出文本"的格式。无论翻译、分类、回归还是问答，都使用文本作为输入和输出。
- **"前缀" (Prefix) 机制**：在输入前添加任务特定的前缀（如"translate English to German: "、"summarize: "、"stsb sentence1: ... sentence2: ..."），告诉模型要执行什么任务。
- **Encoder-Decoder 架构**：T5 使用与原始 Transformer 完全相同的 Encoder-Decoder 架构。编码器双向注意力（类似 BERT），解码器因果注意力（类似 GPT）。
- **C4 数据集 (Colossal Clean Crawled Corpus)**：T5 使用 CommonCrawl 经过严格清洗后得到的 750GB 文本数据。清洗过程包括移除脏话、不完整句子、过长/过短页面等。
- **预训练目标：Span Corruption**：T5 使用"跨度破坏"（Span Corruption）作为预训练目标——随机遮盖文本中的连续多段 token 序列（长度 2-5），模型需要预测被遮盖的序列。相比 MLM（只预测单个 token），Span Corruption 更接近生成任务的需求。
- **Sentinel Token**：使用特殊的哨兵 token 标记被遮盖的跨度位置和顺序，解码器根据哨兵 token 按顺序生成被遮盖的文本。
- **消融研究**：T5 论文包含了大量系统性对比实验（架构、预训练目标、训练策略、数据集等），是 NLP 中最全面的消融研究之一。

## 数学推导

Span Corruption 的预训练目标：
给定输入 $\mathbf{x} = [x_1, \ldots, x_T]$，采样 $k$ 个连续的跨度 $\{s_1, \ldots, s_k\}$，每个跨度长度服从泊松分布（$\lambda=3$），遮盖总长度约 15% 的 token。

输入 $\mathbf{x}_{\text{corrupt}}$ 是 $\mathbf{x}$ 中去掉所有被 span 后再插入哨兵 token 的结果：
$$
\mathbf{x}_{\text{corrupt}} = [\ldots, \langle M_1\rangle, \ldots, \langle M_2\rangle, \ldots]
$$

目标输出是对每个 span 的解码：
$$
\mathbf{y} = [\langle M_1\rangle, s_1, \langle M_2\rangle, s_2, \ldots, \langle M_k\rangle, s_k]
$$

预训练损失：
$$
\mathcal{L} = -\log P(\mathbf{y} | \mathbf{x}_{\text{corrupt}})
$$

## 直观理解

- **Text-to-Text 框架像万能接口**：无论什么任务都通过"文本进、文本出"统一接口处理。就像 USB-C 接口——不论充手机、传文件还是接显示器，都用同一个口。这种统一极大地简化了模型部署和任务迁移。
- **Span Corruption 像句子填空版的"段落默写"**：不是只填单个词，而是要补写一整段被划掉的句子。这样模型学会了"补全长段缺失信息"的能力，对生成任务更友好。
- **C4 数据集像"高质量阅读材料"**：互联网内容良莠不齐，C4 通过严格清洗（去除脏话、无用字符、重复内容）把"垃圾书"过滤掉，留下了高质量的"教科书"。

## 代码示例

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

# 加载 T5 模型
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')

# 1. 翻译任务
trans_prompt = "translate English to Chinese: Natural language processing is fascinating."
inputs = tokenizer(trans_prompt, return_tensors='pt')
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)
print("翻译:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 2. 文本分类（使用文本作为输出标签）
cls_prompt = "sst2 sentence: This movie is incredible!"
inputs = tokenizer(cls_prompt, return_tensors='pt')

# 强制解码——只从候选标签中选择
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_length=10,
        prefix_allowed_tokens_fn=None
    )
print("分类:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 3. 摘要任务
summ_prompt = (
    "summarize: The Transformer architecture has become the dominant approach "
    "in NLP. It uses self-attention mechanisms to process sequences in parallel, "
    "enabling efficient training on large datasets."
)
inputs = tokenizer(summ_prompt, return_tensors='pt', max_length=512, truncation=True)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=30)
print("摘要:", tokenizer.decode(outputs[0], skip_special_tokens=True))

# 4. 预训练 Span Corruption 演示
from transformers import T5ForConditionalGeneration as T5Model
t5_pretrain = T5Model.from_pretrained('t5-small')
inputs = tokenizer("The <extra_id_0> is the <extra_id_1> of language models.",
                   return_tensors='pt')
with torch.no_grad():
    outputs = t5_pretrain.generate(**inputs, max_length=30)
print("Span 预测:", tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 深度学习关联

- **统一文本框架的启发**：T5 的 Text-to-Text 框架启示了后续将多模态任务也统一为"序列到序列"的研究方向（如 Flan-T5、Pix2Seq、Unified-IO）。
- **Encoder-Decoder 的现代应用**：T5 证明了 Encoder-Decoder 架构在大规模预训练场景下的有效性，BART、M2M-100、mT5 等均采用了类似架构。
- **系统化消融研究的榜样**：T5 的论文因其全面系统的消融研究而闻名——对架构、目标函数、训练策略、数据集等进行了上千次实验，为后续研究提供了方法论参考。
