# 6_RAG 与 Agent 的结合

## 1. RAG 回顾与 Agent 化

传统 RAG 是"一次检索+一次生成"的流水线，而 Agent 化的 RAG 能够**自主决定何时检索、检索什么、如何使用检索结果**。

```
传统 RAG 流程:
  Query → Retrieve → Generate → Answer
  (固定流程，单次执行)

Agent-RAG 流程:
  Query → LLM 决策 → [检索 | 直接回答 | 先分析再检索] → 迭代 → Answer
  (动态决策，多轮执行)
```

## 2. 检索作为工具

```python
class RAGTools:
    """将检索能力封装为 Agent 工具"""

    def __init__(self, vector_store, embedding_model):
        self.vstore = vector_store
        self.embed = embedding_model

    def search_documents(self, query: str, top_k: int = 5,
                         source: str = "all") -> str:
        """搜索知识库文档"""
        query_embedding = self.embed.encode(query)

        filter_dict = None
        if source != "all":
            filter_dict = {"source": source}

        results = self.vstore.search(
            query_embedding, top_k=top_k, filter=filter_dict
        )

        output = []
        for doc, score in results:
            output.append(
                f"[相关度: {score:.2f}] 来源: {doc.metadata['source']}\n"
                f"内容: {doc.page_content}\n"
            )

        return "\n---\n".join(output) if output else "未找到相关文档"

    def search_with_filters(self, query: str, date_range: tuple = None,
                            doc_type: str = None) -> str:
        """带过滤条件的高级搜索"""
        filters = {}
        if date_range:
            filters["date"] = {"$gte": date_range[0], "$lte": date_range[1]}
        if doc_type:
            filters["doc_type"] = doc_type

        return self.search_documents(query, filter_dict=filters)

# 注册为 Agent 工具
agent_tools = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "搜索公司内部知识库。当问题涉及公司产品、政策、技术文档时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索查询"},
                    "source": {
                        "type": "string",
                        "enum": ["all", "product", "policy", "tech"],
                        "description": "限定搜索范围"
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

## 3. 动态检索策略

### 3.1 自适应检索决策

```python
class AdaptiveRAGAgent:
    """根据问题复杂度动态决定检索策略"""

    def __init__(self, llm, rag_tools):
        self.llm = llm
        self.rag = rag_tools

    def answer(self, question: str) -> str:
        # 步骤 1: 判断是否需要检索
        decision = self.llm.generate(f"""
判断以下问题是否需要检索外部知识库来回答。

问题: {question}

输出 JSON:
{{"need_retrieve": true/false, "reason": "原因", "search_queries": ["查询1", "查询2"]}}
""")
        decision = json.loads(decision)

        if not decision["need_retrieve"]:
            # 模型直接回答
            return self.llm.generate(question)

        # 步骤 2: 执行检索
        all_context = []
        for query in decision["search_queries"]:
            results = self.rag.search_documents(query)
            all_context.append(results)

        # 步骤 3: 基于检索结果生成答案
        context = "\n\n".join(all_context)
        return self.llm.generate(f"""
基于以下参考资料回答问题。如果资料不足，说明哪些信息缺失。

参考资料:
{context}

问题: {question}
""")
```

### 3.2 多步检索 (Iterative Retrieval)

```python
class IterativeRAGAgent:
    """多轮迭代检索 Agent"""

    def __init__(self, llm, rag_tools, max_iterations: int = 3):
        self.llm = llm
        self.rag = rag_tools
        self.max_iter = max_iterations

    def answer(self, question: str) -> str:
        gathered_info = []
        current_understanding = question

        for i in range(self.max_iter):
            # 生成当前轮次的检索查询
            search_query = self.llm.generate(f"""
原始问题: {question}
已收集信息: {gathered_info}
当前理解: {current_understanding}

还需要什么信息来完整回答问题？
生成一个精准的搜索查询。如果信息已足够，输出 "SUFFICIENT"。
""")

            if "SUFFICIENT" in search_query:
                break

            # 执行检索
            results = self.rag.search_documents(search_query, top_k=3)
            gathered_info.append({
                "query": search_query,
                "results": results,
                "iteration": i + 1
            })

            # 更新理解
            current_understanding = self.llm.generate(f"""
基于新获取的信息，更新对问题的理解：

原始问题: {question}
新信息: {results}
""")

        # 最终生成
        return self.generate_final_answer(question, gathered_info)
```

## 4. 检索结果质量评估

```python
class RetrievalEvaluator:
    """评估检索结果的相关性和质量"""

    def evaluate(self, query: str, retrieved_docs: list[dict],
                 llm) -> dict:
        scores = []
        for doc in retrieved_docs:
            eval_result = llm.generate(f"""
评估以下检索文档与查询的相关性。

查询: {query}
文档内容: {doc['content'][:500]}

评分标准：
- 0: 完全不相关
- 0.5: 部分相关
- 1.0: 高度相关

输出 JSON: {{"score": X.X, "reason": "评分原因"}}
""")
            scores.append(json.loads(eval_result))

        avg_score = sum(s["score"] for s in scores) / len(scores)
        return {
            "avg_relevance": avg_score,
            "needs_refinement": avg_score < 0.5,
            "details": scores
        }

class QualityAwareRAGAgent:
    """根据检索质量调整策略"""

    def answer(self, question: str) -> str:
        # 初始检索
        docs = self.rag.search_documents(question)

        # 评估质量
        quality = self.evaluator.evaluate(question, docs, self.llm)

        if quality["needs_refinement"]:
            # 检索质量差，尝试改写查询
            refined_query = self.llm.generate(f"""
原始查询 '{question}' 检索效果不好。
请生成一个更精准的搜索查询。
""")
            docs = self.rag.search_documents(refined_query)

        return self.generate_answer(question, docs)
```

## 5. 多源检索融合

```python
class MultiSourceRAGAgent:
    """从多个数据源检索并融合"""

    def __init__(self, llm):
        self.llm = llm
        self.sources = {}

    def add_source(self, name: str, retriever, description: str):
        self.sources[name] = {
            "retriever": retriever,
            "description": description
        }

    def answer(self, question: str) -> str:
        # 选择数据源
        source_selection = self.llm.generate(f"""
问题: {question}

可用数据源:
{self._format_sources()}

选择最适合的数据源（可多选），输出名称列表。
""")
        selected = self.parse_sources(source_selection)

        # 并行检索
        all_results = {}
        for source_name in selected:
            results = self.sources[source_name]["retriever"](question)
            all_results[source_name] = results

        # 融合结果
        return self.fuse_and_answer(question, all_results)

    def fuse_and_answer(self, question: str, results: dict) -> str:
        combined = ""
        for source, docs in results.items():
            combined += f"\n=== 来自 {source} ===\n"
            for doc in docs:
                combined += f"{doc}\n"

        return self.llm.generate(f"""
基于以下多源信息回答问题。标注信息来源。

{combined}

问题: {question}
""")
```

## 6. RAG + Agent 实战模式

### 6.1 文档问答 Agent

```python
class DocumentQAAgent:
    """智能文档问答"""

    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vstore = vector_store

        self.tools = {
            "search": self.search,
            "read_section": self.read_section,
            "compare_documents": self.compare_documents,
        }

    def search(self, query: str) -> str:
        """搜索相关文档片段"""
        results = self.vstore.similarity_search(query, k=5)
        return "\n---\n".join(
            f"[{doc.metadata['source']} - 第{doc.metadata.get('page', '?')}页]\n"
            f"{doc.page_content}"
            for doc in results
        )

    def read_section(self, doc_id: str, section: str) -> str:
        """读取文档的特定章节"""
        results = self.vstore.search_with_filter(
            {"doc_id": doc_id, "section": section}
        )
        return "\n".join(doc.page_content for doc in results)

    def compare_documents(self, doc_ids: list[str], aspect: str) -> str:
        """对比多个文档在某个方面的异同"""
        docs_content = {}
        for did in doc_ids:
            docs_content[did] = self.read_section(did, aspect)

        return self.llm.generate(f"对比以下文档在'{aspect}'方面的异同：\n{docs_content}")
```

### 6.2 带引用的 RAG

```python
class CitedRAGAgent:
    """生成带来源引用的答案"""

    def answer_with_citations(self, question: str) -> dict:
        # 检索
        docs = self.vstore.similarity_search_with_score(question, k=5)

        # 生成带引用的答案
        context_with_ids = "\n".join(
            f"[文档{i+1}] (来源: {doc.metadata['source']})\n{doc.page_content}"
            for i, (doc, score) in enumerate(docs)
        )

        answer = self.llm.generate(f"""
基于以下文档回答问题，并在答案中引用文档编号。

文档:
{context_with_ids}

问题: {question}

回答时使用 [文档X] 格式标注信息来源。
""")

        # 解析引用
        citations = self.extract_citations(answer, docs)

        return {
            "answer": answer,
            "sources": citations,
            "confidence": self.calculate_confidence(docs)
        }
```

## 7. RAG 性能优化

```python
# 优化 1: 查询改写
class QueryRewriter:
    def rewrite(self, query: str) -> list[str]:
        """将用户查询改写为更有效的检索查询"""
        prompt = f"""
将用户问题改写为 2-3 个不同角度的搜索查询。

用户问题: {query}

输出 JSON 数组: ["查询1", "查询2", "查询3"]
"""
        return json.loads(self.llm.generate(prompt))

# 优化 2: 检索结果重排序
class Reranker:
    def rerank(self, query: str, docs: list, top_k: int = 3) -> list:
        """用交叉编码器重排序"""
        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.cross_encoder.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

# 优化 3: 分块策略
class SmartChunking:
    def chunk(self, document: str, strategy: str = "semantic") -> list[str]:
        if strategy == "semantic":
            # 按语义分块
            return self.semantic_chunker.split(document)
        elif strategy == "recursive":
            # 递归字符分割
            return self.recursive_splitter.split(document)
        elif strategy == "parent_child":
            # 父子分块：大块检索，小块返回
            return self.parent_child_chunker.split(document)
```

## 总结

RAG 与 Agent 的结合核心在于**将"固定流水线"变为"智能决策循环"**。Agent 能自主决定何时检索、评估检索质量、必要时改写查询并重新检索。这种动态性大幅提升了复杂问题的处理能力，但也增加了延迟和成本，需要在质量和效率之间找到平衡。
