# 7_LangChain Agent 架构

## 1. LangChain Agent 概述

LangChain 是最流行的 LLM 应用开发框架之一，其 Agent 模块提供了**将 LLM 与工具结合**的标准化方式。

```
LangChain Agent 架构:

┌─────────────────────────────────────────────┐
│              AgentExecutor                    │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐ │
│  │  Agent   │→│  Action  │→│ Observation│ │
│  │ (LLM决策)│  │ (工具调用) │  │ (结果返回) │ │
│  └──────────┘  └──────────┘  └───────────┘ │
│       ↑                               │      │
│       └───────────── 循环 ─────────────┘      │
│                                              │
│  工具集: [Tool₁, Tool₂, ..., Toolₙ]         │
│  记忆:   ConversationBufferMemory            │
└─────────────────────────────────────────────┘
```

## 2. Tool 类定义

### 2.1 基础 Tool

```python
from langchain.tools import Tool, tool
from langchain.tools.base import BaseTool
from typing import Optional, Type
from pydantic import BaseModel, Field

# 方式 1: 使用 @tool 装饰器
@tool
def search_web(query: str) -> str:
    """搜索互联网获取信息。当需要查询实时信息时使用。"""
    # 实际实现
    return f"搜索 '{query}' 的结果..."

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入应为有效的数学表达式字符串。"""
    try:
        result = eval(expression)  # 生产环境请使用安全的表达式解析器
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

# 方式 2: 使用 Pydantic 定义参数 schema
class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")
    num_results: int = Field(default=5, description="返回结果数量")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "搜索互联网获取最新信息"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str, num_results: int = 5) -> str:
        # 实际搜索逻辑
        return f"搜索 '{query}' 的前 {num_results} 条结果..."

    async def _arun(self, query: str, num_results: int = 5) -> str:
        # 异步版本
        return self._run(query, num_results)
```

### 2.2 工具集 (ToolKit)

```python
from langchain.agents import Tool

# 组织相关工具
data_analysis_tools = [
    Tool(
        name="query_database",
        func=db_query,
        description="执行 SQL 查询。输入应为有效的 SQL 语句。"
    ),
    Tool(
        name="generate_chart",
        func=create_chart,
        description="根据数据生成图表。输入应为 JSON 格式的数据和图表类型。"
    ),
    Tool(
        name="export_csv",
        func=export_to_csv,
        description="将数据导出为 CSV 文件。输入应为查询结果。"
    ),
]
```

## 3. Agent 类型

### 3.1 ReAct Agent (Zero-shot)

```python
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain import hub

llm = ChatOpenAI(model="gpt-4", temperature=0)
tools = [search_web, calculator]

# 从 LangChain Hub 拉取 ReAct 提示词
prompt = hub.pull("hwchase17/react")

# 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,          # 打印执行过程
    max_iterations=10,     # 最大迭代次数
    handle_parsing_errors=True,  # 处理解析错误
    return_intermediate_steps=True  # 返回中间步骤
)

# 执行
result = agent_executor.invoke({"input": "2024年GDP最高的国家是哪个？"})
```

### 3.2 OpenAI Tools Agent (Function Calling)

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent

llm = ChatOpenAI(model="gpt-4", temperature=0)
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "分析最近一个月的销售数据趋势"})
```

### 3.3 Structured Chat Agent

```python
from langchain.agents import AgentExecutor, create_structured_chat_agent

# 适用于需要多参数工具的场景
agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

## 4. 记忆集成

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
    VectorStoreRetrieverMemory
)

# 基础缓冲记忆
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# 摘要记忆 — 自动压缩长对话
summary_memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history"
)

# 滑动窗口记忆 — 只保留最近 N 轮
window_memory = ConversationBufferWindowMemory(
    k=5,  # 保留最近 5 轮
    memory_key="chat_history"
)

# 向量存储记忆 — 长期记忆
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts([], embedding_model)
vector_memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    memory_key="relevant_history"
)

# 组合使用
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# 多轮对话
response1 = agent_executor.invoke({"input": "我叫张三，喜欢Python"})
response2 = agent_executor.invoke({"input": "我叫什么名字？"})
# Agent 会记住用户的名字
```

## 5. 自定义 Agent

```python
from langchain.agents import AgentExecutor, BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish

class CustomAgent(BaseSingleActionAgent):
    """自定义 Agent 实现"""

    @property
    def input_keys(self):
        return ["input"]

    def plan(self, intermediate_steps, **kwargs):
        """决定下一步行动"""
        # 构建历史
        history = self._format_steps(intermediate_steps)

        # LLM 决策
        response = llm.generate(f"""
可用工具: {self._format_tools()}
历史: {history}
当前问题: {kwargs['input']}

决策 (JSON):
{{"action": "工具名或 final_answer", "action_input": "参数或最终答案"}}
""")

        decision = json.loads(response)

        if decision["action"] == "final_answer":
            return AgentFinish(
                return_values={"output": decision["action_input"]},
                log=response
            )

        return AgentAction(
            tool=decision["action"],
            tool_input=decision["action_input"],
            log=response
        )

    async def aplan(self, intermediate_steps, **kwargs):
        return self.plan(intermediate_steps, **kwargs)

# 使用
custom_agent = CustomAgent()
executor = AgentExecutor(agent=custom_agent, tools=tools, verbose=True)
```

## 6. 链式 Agent (Agent Chains)

```python
from langchain.chains import LLMChain, SequentialChain

# 路由 Agent：根据问题类型分发到不同 Agent
from langchain.agents import AgentType, initialize_agent

research_agent = initialize_agent(
    research_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

analysis_agent = initialize_agent(
    analysis_tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# 路由链
from langchain.chains.router import MultiRouteChain, RouterChain

class TaskRouter:
    def __init__(self, llm, agents: dict):
        self.llm = llm
        self.agents = agents

    def route(self, task: str) -> str:
        # LLM 判断任务类型
        category = self.llm.generate(f"""
将任务分类到以下类别之一: {list(self.agents.keys())}
任务: {task}
类别:
""")
        agent = self.agents.get(category.strip(), self.agents["general"])
        return agent.invoke({"input": task})
```

## 7. 流式输出

```python
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 流式 Agent
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=[StreamingStdOutCallbackHandler()],
    verbose=False  # 关闭 verbose 以使用自定义回调
)

# 异步流式
async def stream_agent():
    async for event in agent_executor.astream_events(
        {"input": "分析2024年AI行业趋势"},
        version="v1"
    ):
        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"].content, end="", flush=True)
        elif event["event"] == "on_tool_start":
            print(f"\n[调用工具: {event['name']}]")
        elif event["event"] == "on_tool_end":
            print(f"[工具返回: {event['data'].output[:100]}...]")
```

## 8. 错误处理与容错

```python
from langchain.agents import AgentExecutor

class RobustAgentExecutor(AgentExecutor):
    """增强容错的 Agent 执行器"""

    def _take_next_step(self, name_to_tool_map, color_mapping, inputs, intermediate_steps):
        try:
            return super()._take_next_step(
                name_to_tool_map, color_mapping, inputs, intermediate_steps
            )
        except Exception as e:
            # 工具调用失败时的降级策略
            error_msg = f"工具调用失败: {str(e)}。请尝试其他方法或直接回答。"

            # 将错误信息注入上下文
            intermediate_steps.append(
                (AgentAction("error_handler", error_msg, ""), error_msg)
            )
            return intermediate_steps

# 配置
agent_executor = RobustAgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    max_execution_time=60,  # 超时限制
    handle_parsing_errors=True,
    early_stopping_method="generate"  # 超限时让 LLM 总结已有结果
)
```

## 9. LangChain vs LangGraph 趋势

```
发展趋势:

LangChain Agent          →        LangGraph
(2023)                          (2024+)

- 线性执行循环              - 图结构工作流
- 有限状态管理              - 完整状态管理
- 简单 if/else 控制         - 条件分支+循环
- 单 Agent 为主             - 多 Agent 编排

建议:
- 简单 Agent 任务 → LangChain AgentExecutor 足够
- 复杂工作流 → 迁移到 LangGraph（见第8章）
- 新项目 → 直接使用 LangGraph
```

## 总结

LangChain Agent 提供了**标准化的 Agent 开发范式**，核心组件是 Tool、Agent 和 AgentExecutor。优势在于生态丰富、上手快；局限在于复杂工作流控制能力有限。对于需要复杂状态管理的场景，建议直接使用 LangGraph。
