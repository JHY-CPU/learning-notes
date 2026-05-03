# 43_检索增强生成 (RAG) 架构设计

## 核心概念

- **RAG (Retrieval-Augmented Generation)**：将信息检索与文本生成相结合的方法。在模型生成回答前，先从外部知识库中检索相关文档片段，作为生成的上下文信息。由 Lewis et al. (2020) 提出。
- **RAG 的核心架构**：包含三个主要阶段——检索（Retrieval）、增强（Augmentation）、生成（Generation）。输入查询 -> 检索相关文档 -> 拼接为 prompt -> LLM 生成回答。
- **检索器 (Retriever)**：将输入查询编码为向量（query embedding），在向量数据库中搜索最相似的文档片段（top-k 检索）。常用模型有 DPR、Contriever、Sentence-BERT。
- **生成器 (Generator)**：使用 LLM（如 GPT、LLaMA、T5）基于检索到的文档和原始查询生成回答。生成器可以是任意条件文本生成模型。
- **端到端训练 vs 组件化**：原始 RAG 论文对整个 pipeline（检索器 + 生成器）端到端训练，但实践中通常使用"冻检索 + 冻生成"的组件化方案，分别使用现成的检索器和 LLM。
- **知识更新**：RAG 的核心优势——更新知识库即可更新模型的知识，无需重新训练模型。这解决了 LLM 知识截止的问题。
- **Chunking 策略**：文档需要切分成适当大小的块（chunk），大小决定检索粒度。块太小导致上下文不足，块太大包含过多噪声。常用 256-1024 token。
- **RAG 的变体**：
  - Naive RAG：检索 + 生成的基本流程
  - Advanced RAG：引入查询重写、检索后重排序等优化
  - Modular RAG：灵活的模块化编排（如 LangChain 的实现）

## 数学推导

RAG 的条件生成概率：
$$
P(y | x) \propto \sum_{z \in \text{top-k}(q, D)} P_{\text{retriever}}(z | x) \cdot P_{\text{generator}}(y | x, z)
$$

实际实现中通常简化（直接使用检索结果作为上下文）：
$$
y = \arg\max P_{\text{LLM}}(y | \text{Prompt}(x, z_1, z_2, \ldots, z_k))
$$

**检索相似度计算**：
$$
\text{sim}(q, d) = \frac{E_Q(q)^\top E_D(d)}{\|E_Q(q)\| \cdot \|E_D(d)\|}
$$

其中 $E_Q$ 和 $E_D$ 分别是 Query 编码器和 Document 编码器（可以是同一个或不同的编码器）。

## 直观理解

- **RAG 像"开卷考试"**：传统 LLM 像闭卷考试——只能靠记忆（训练数据）来回答问题。RAG 是开卷考试——先查阅参考资料（检索），然后结合参考资料和问题作答。结果不仅更准确，还能引用原文来源。
- **检索 + 生成的双轮驱动**：检索器像图书馆管理员——快速找到最相关的几本书。生成器像一位学者——阅读这些书的内容后给出综合回答。两者互相配合，缺一不可。
- **RAG 像"给模型配外脑"**：模型的大脑（参数）是固定的，但模型可以随时查阅外部存储器（向量数据库）。这样既保留了模型的语言能力，又获得了实时更新的知识。
- **Chunking 像"图书分类"**：将一本书分成若干章节（chunks）存放在图书馆中。太细的划分（按段落）让你容易丢失前后文，太粗的划分（整本书）让你难以精确定位。

## 代码示例

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import numpy as np

# 模拟 RAG 的简化实现

class SimpleRAG:
    """简化的 RAG 系统"""
    def __init__(self, llm_name='t5-small'):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModelForSeq2SeqLM.from_pretrained(llm_name)
        # 模拟文档库
        self.documents = [
            "上海是中国最大的城市之一，位于长江入海口。",
            "Transformer 是一种基于自注意力机制的神经网络架构。",
            "自然语言处理是人工智能的重要分支。",
            "BERT 是由 Google 开发的预训练语言模型。",
            "注意力机制允许模型在生成每个词时关注输入的不同位置。",
        ]
        # 模拟文档嵌入（实际中由 Sentence-BERT 等生成）
        self.doc_embeddings = np.random.randn(len(self.documents), 128)

    def retrieve(self, query, k=2):
        """模拟检索过程"""
        query_emb = np.random.randn(128)  # 实际中需要编码
        scores = [np.dot(query_emb, de) for de in self.doc_embeddings]
        top_k = np.argsort(scores)[-k:][::-1]
        return [self.documents[i] for i in top_k]

    def generate(self, query):
        # 1. 检索
        relevant_docs = self.retrieve(query)
        # 2. 构建 prompt
        context = "\n".join(relevant_docs)
        prompt = f"基于以下信息回答问题。\n信息：{context}\n问题：{query}\n答案："
        # 3. 生成
        inputs = self.tokenizer(prompt, return_tensors='pt', max_length=512, truncation=True)
        outputs = self.llm.generate(**inputs, max_length=50)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

# RAG 流程演示
rag = SimpleRAG()
query = "什么是 Transformer？"
answer = rag.generate(query)
print(f"问题: {query}")
print(f"检索到: {rag.retrieve(query)}")
print(f"生成回答: {answer}")

# RAG 的实用技巧
print("\n--- RAG 优化技巧 ---")
print("1. 查询重写: 将用户问题改写为更适合检索的形式")
print("2. 重排序: 对检索结果进行语义重排序提升质量")
print("3. 上下文压缩: 压缩检索结果去除噪声")
print("4. 检索融合: 同时使用稀疏检索(BM25)和稠密检索")
```

## 深度学习关联

- **解决 LLM 知识固化问题**：RAG 是解决 LLM 知识截止、事实幻觉和领域定制的主流方案，在 ChatGPT 插件、Bing Chat、Google Bard 等产品中广泛应用。
- **与微调互补**：RAG 和微调（包括 LoRA）解决不同问题——RAG 用于引入外部知识，微调用于改变模型的行为和输出风格。两者可以结合使用。
- **多模态 RAG**：RAG 思想正在向多模态扩展——检索图像来帮助回答视觉问题（如 REVEAL）、检索音频/视频来辅助多模态生成。
