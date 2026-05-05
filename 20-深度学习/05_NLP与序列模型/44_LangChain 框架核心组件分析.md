# 45_LangChain 框架核心组件分析

## 核心概念

- **LangChain**：由 Harrison Chase 于 2022 年创建的开源框架，用于开发基于 LLM 的应用。提供了一套模块化工具链，将 LLM 与外部的数据源、API、工具等连接起来。
- **核心组件体系**：
  - **Models (模型)**：LLM 的抽象接口，支持 OpenAI、Anthropic、HuggingFace、本地模型等
  - **Prompts (提示)**：模板化 prompt 管理，支持变量注入、示例选择、输出解析
  - **Chains (链)**：将多个步骤串联成工作流（如 "检索 -> 提示 -> 生成 -> 解析"）
  - **Agents (代理)**：让 LLM 自主决定调用哪些工具、以什么顺序执行
  - **Memory (记忆)**：维护对话历史的持久化存储
  - **Indexes (索引)**：文档加载、切分、向量化、检索的整体方案
- **LCEL (LangChain Expression Language)**：声明式的链组合语法，使用 `|` 运算符将组件串联，如 `prompt | model | output_parser`。
- **工具 (Tools)**：LLM 可以调用的外部函数（搜索、计算器、数据库查询等）。工具的描述被作为系统 prompt 提供给 LLM。
- **Agent 的 ReAct 模式**：结合推理（Reasoning）和行动（Acting）的 Agent 范式——LLM 逐步思考"需要做什么"->"执行工具"->"观察结果"->"继续推理"。
- **回调系统 (Callbacks)**：提供对链执行过程的钩子（hooks），用于日志记录、监控、流式输出等。

## 数学推导

LangChain 的核心逻辑是计算图（Computational Graph）的执行。

**Chain 的前向传播**：
$$
\text{Chain}(x) = f_N \circ f_{N-1} \circ \ldots \circ f_1(x)
$$

每个 $f_i$ 可以是一个模型调用、一个检索操作、一个 prompt 格式化等。

**Agent 的 ReAct 循环**（收敛性）：
$$
\text{Agent}(x) = \text{Loop}\left(\text{Think} \to \text{Act} \to \text{Observe} \to \text{Continue or Stop}\right)
$$

直到 Agent 产生 Final Answer 或达到最大迭代次数。

## 直观理解

- **LangChain 像"LLM 应用的乐高积木"**：提供了一系列标准化的积木块（Model、Prompt、Chain、Agent 等），你可以像搭积木一样自由组合，构建复杂的 LLM 应用，而不需要从零开始写胶水代码。
- **Chain 像"流水线"**：一个 Chain 就像工厂的流水线——原材料（用户输入）经过一系列处理步骤（格式化提示 -> 调用 LLM -> 解析输出 -> 处理结果），最终产出成品（回答）。
- **Agent 像"有自主权的员工"**：Agent 不是简单的"输入-输出"流水线，而是自己能做出决策——它接收一个任务，自己规划需要哪些工具、什么步骤、如何验证结果。
- **LCEL 的 | 运算符就像 Unix 管道**：在 Unix 中 `cat file | grep keyword | sort` 将文本处理串联，LCEL 中 `prompt | model | parser` 将 LLM 处理步骤串联。

## 代码示例

```python
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# 使用 LCEL 构建 RAG 链
# 注意：需要安装 langchain, langchain-openai 等包

# 模拟的简单链（无需 API 密钥）
from langchain.llms.fake import FakeListLLM

# 1. Prompt 模板
prompt = ChatPromptTemplate.from_template(
    "基于以下信息回答问题:\n\n信息: {context}\n\n问题: {question}\n\n回答:"
)

# 2. 模拟 LLM
fake_llm = FakeListLLM(responses=["LLM 是基于 Transformer 的模型。"])

# 3. 输出解析器
output_parser = StrOutputParser()

# 4. 模拟检索函数
def retrieve_docs(question):
    return f"与'{question}'相关的文档内容。"

# 5. LCEL 链
chain = (
    {"context": retrieve_docs, "question": RunnablePassthrough()}
    | prompt
    | fake_llm
    | output_parser
)

result = chain.invoke("什么是 LLM？")
print(f"Chain 执行结果: {result}")

# Agent 模式示例（使用工具）
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool

@tool
def search(query: str) -> str:
    """搜索网络信息"""
    return f"关于'{query}'的搜索结果"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [search, calculator]
print(f"\n已注册工具: {[t.name for t in tools]}")
print("Agent 会自主选择使用哪个工具完成任务。")
```

## 深度学习关联

- **LLM 应用的标准框架**：LangChain 已成为构建 LLM 应用的事实标准之一，与 LlamaIndex、Haystack 等框架竞争。其模块化设计降低了开发门槛。
- **RAG 应用的构建器**：LangChain 提供了从文档加载、切分、向量化到检索、生成的完整 RAG 工具链，是快速原型开发的首选。
- **生态系统的演进**：LangChain 带动了整个 LLM 工具链生态，包括 LangSmith（调试和监控）、LangServe（部署）、LangHub（共享链和 prompt），成为越来越完整的 LLMOps 平台。
