# LangChain与框架对比 - Agent与工具调用

*深入理解 LangChain Chains/Agents/Tools 核心概念，对比 LlamaIndex、Semantic Kernel 等主流 LLM 应用框架，明确各自适用场景*

四大 LLM 应用框架对比

| 维度 | LangChain | LlamaIndex | Semantic Kernel | Haystack (deepset) |
| --- | --- | --- | --- | --- |
| **定位** | 通用 LLM 应用框架 | RAG 专用 | 企业级 LLM 框架 | NLP Pipeline 框架 |
| **主语言** | Python/JS | Python | C#/Python | Python |
| **学习曲线** | 中-高 | 低-中 | 中 | 中 |
| **RAG 能力** | 中（需组合） | 最强 | 中 | 强 |
| **Agent 能力** | 最强 | 基础 | 中 | 中 |
| **生态/集成** | 最丰富 | 丰富 | 微软生态 | 中等 |
| **状态管理** | LangGraph | 基础 | 内置 | Pipeline |
| **可观测性** | LangSmith | LlamaTrace | Azure Monitor | deepset Cloud |
| **稳定性** | 迭代快，API 变化多 | 较稳定 | 稳定 | 稳定 |

场景选型指南

| 需求场景 | 推荐框架 | 理由 |
| --- | --- | --- |
| 构建复杂 Agent 系统 | LangChain + LangGraph | Agent 和状态机能力最强 |
| 纯 RAG 应用 | LlamaIndex | RAG 专用，开箱即用 |
| .NET/企业级应用 | Semantic Kernel | 微软生态深度集成 |
| 生产级 NLP Pipeline | Haystack | Pipeline 抽象成熟 |
| 快速原型验证 | LangChain（LCEL） | 生态最丰富，组合灵活 |
| 轻量级简单场景 | 直接调用 API | 避免框架抽象的开销 |

## LangChain 核心架构

### LCEL (LangChain Expression Language)

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# LCEL 管道式组合
chain = prompt | llm | StrOutputParser()

# 流式输出
async for chunk in chain.astream({"topic": "量子计算"}):
    print(chunk, end="")

# 并行执行多个链
from langchain_core.runnables import RunnableParallel
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
)
result = parallel.invoke({"doc": document})
```

### LangGraph 状态机

```python
from langgraph.graph import StateGraph, END

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", call_tools)
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    END: END,
})
workflow.add_edge("tools", "agent")
workflow.set_entry_point("agent")
app = workflow.compile()
```

## LlamaIndex 核心特性

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# 简单 RAG
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("什么是量子纠缠？")

# 高级：自定义检索策略
from llama_index.core.retrievers import AutoMergingRetriever
retriever = AutoMergingRetriever(
    index.as_retriever(similarity_top_k=5),
    storage_context=index.storage_context,
)
```

## 框架选型决策树

```
需要 LLM 应用框架？
│
├── 需要复杂 Agent/工作流？
│   └── LangChain + LangGraph
│
├── 主要是 RAG？
│   ├── 需要高级索引策略？→ LlamaIndex
│   └── 简单 RAG → LangChain 或直接 API
│
├── .NET/企业级/C# 生态？
│   └── Semantic Kernel
│
├── 生产级 NLP Pipeline？
│   └── Haystack
│
└── 最小化依赖/学习成本？
    └── 直接调用 OpenAI/Claude API
```

## 无框架方案对比

```python
# 直接调用 API（最灵活，无抽象开销）
import anthropic
client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[{"role": "user", "content": "..."}],
)

# 优势：完全控制、无锁定、调试简单
# 劣势：需要自行管理状态、工具调用、重试逻辑
```


<!-- Converted from: 02_LangChain与框架对比.html -->
