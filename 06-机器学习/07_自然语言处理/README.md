# 自然语言处理

## 一、文本表示

### 1.1 传统方法

- **One-Hot**：高维稀疏，无法表示语义相似度
- **TF-IDF**：词频-逆文档频率，衡量词的重要性
- **Bag of Words**：忽略词序，统计词频

### 1.2 词嵌入

- **Word2Vec**：Skip-gram / CBOW，通过上下文学习词向量
- **GloVe**：全局共现矩阵分解
- **FastText**：子词级别，支持OOV

关键特性：词向量空间中语义相似的词距离近。

---

## 二、NLP任务

### 2.1 文本分类

- 情感分析、垃圾邮件检测、新闻分类
- 方法：CNN、RNN、BERT fine-tuning

### 2.2 序列标注

- **NER（命名实体识别）**：识别人名、地名、组织名等
- **POS Tagging**：词性标注
- **方法**：BiLSTM-CRF、BERT-CRF

### 2.3 机器翻译

- 统计机器翻译（SMT）
- 神经机器翻译（NMT）：Seq2Seq + Attention
- Transformer：完全基于注意力

### 2.4 问答系统

- 抽取式问答：从段落中抽取答案
- 生成式问答：生成自由形式答案
- RAG：检索增强生成

---

## 三、大语言模型时代

### 3.1 Prompt Engineering

- Zero-shot、Few-shot prompting
- Chain-of-Thought推理
- In-Context Learning

### 3.2 微调

- Full Fine-tuning
- LoRA / QLoRA：低秩适配，参数高效微调
- Prefix Tuning、Adapter

### 3.3 对齐

- SFT → RLHF → DPO
- Constitutional AI
