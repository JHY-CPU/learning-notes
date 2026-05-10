# 高级RAG架构 - RAG与向量数据库

*GraphRAG、Self-RAG、CRAG、多模态 RAG 及高级分块与评估策略*

ColBERT vs 传统方案对比

| 方案 | 交互方式 | 速度 | 精度 |
| --- | --- | --- | --- |
| Bi-Encoder（Dense Retrieval） | 向量点积 | 最快 | 中 |
| Cross-Encoder（Reranker） | 全交叉注意力 | 最慢 | 最高 |
| ColBERT | MaxSim 延迟交互 | 快 | 高 |


ColBERT 为每个 token 生成嵌入向量，查询时通过 MaxSim（最大相似度）计算相关性，兼顾速度与精度。

多模态 RAG 方案对比

| 方案 | 原理 | 优点 | 缺点 |
| --- | --- | --- | --- |
| **文本提取法** | 将图表/表格转为文本描述 | 兼容传统 RAG | 丢失视觉信息 |
| **多模态嵌入法** | 使用 CLIP 等统一编码图文 | 保留视觉语义 | 文本精度下降 |
| **MM-LLM 直接理解** | 用 GPT-4V/Gemini 理解文档图像 | 理解力最强 | 成本高、速度慢 |
| **混合方案** | 文本走 RAG，图像走 MM-LLM | 兼顾精度和成本 | 架构复杂 |

RAGAS 评估体系详解

| 指标 | 评估对象 | 计算方法 | 理想值 |
| --- | --- | --- | --- |
| **Context Precision** | 检索排序质量 | 相关文档在检索结果中的排名 | 1.0 |
| **Context Recall** | 检索覆盖度 | 标准答案所需信息被检索到的比例 | 1.0 |
| **Faithfulness** | 生成忠实度 | 回答中可被上下文支持的句子比例 | 1.0 |
| **Answer Relevancy** | 回答相关性 | 生成回答与问题的相关程度（LLM 评分） | 1.0 |
| **Answer Correctness** | 回答正确性 | 与标准答案的语义和事实对比 | 1.0 |
| **Answer Semantic Similarity** | 语义相似度 | 回答与标准答案的嵌入相似度 | 1.0 |


<!-- Converted from: 03_高级RAG架构.html -->
