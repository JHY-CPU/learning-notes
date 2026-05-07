# 8_LangGraph：有状态 Agent 编排

## 1. LangGraph 概述

LangGraph 是 LangChain 团队开发的**有状态图编排框架**，专为构建复杂的多步 Agent 工作流设计。它将 Agent 行为建模为**有向图**，支持条件分支、循环和人机协作。

```
LangGraph vs LangChain AgentExecutor:

AgentExecutor:           LangGraph:
  线性循环                 图结构
  ┌───┐                  ┌───┐
  │LLM│                  │ A │
  └─┬─┘                  └─┬─┘
    ↓                    ╱   ╲
  ┌───┐              ┌───┐ ┌───┐
  │Tool│              │ B │ │ C │
  └─┬─┘              └─┬─┘ └─┬─┘
    ↓                    ╲   ╱
  ┌───┐                  ┌───┐
  │LLM│                  │ D │
  └───┘                  └───┘
  简单但受限              灵活且强大
```

## 2. 核心概念

### 2.1 StateGraph（状态图）

```python
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
import operator

# 定义状态 Schema
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # 消息累加
    current_plan: str
    step_count: int
    tools_used: list[str]
    final_answer: str

# 创建状态图
graph = StateGraph(AgentState)
```

### 2.2 Node（节点）

```python
# 节点是处理函数，接收状态，返回状态更新
def planner_node(state: AgentState) -> dict:
    """规划节点：分析问题，制定计划"""
    messages = state["messages"]
    response = llm.invoke(messages)

    return {
        "current_plan": response.content,
        "step_count": state["step_count"] + 1
    }

def executor_node(state: AgentState) -> dict:
    """执行节点：调用工具"""
    plan = state["current_plan"]
    tool_call = parse_tool_call(plan)

    result = tools[tool_call["name"]](**tool_call["args"])

    return {
        "messages": [AIMessage(content=f"工具结果: {result}")],
        "tools_used": state["tools_used"] + [tool_call["name"]]
    }

def reviewer_node(state: AgentState) -> dict:
    """审查节点：检查结果是否满足需求"""
    review = llm.invoke(
        f"检查以下信息是否足以回答问题:\n{state['messages'][-1].content}"
    )
    return {"messages": [AIMessage(content=review.content)]}

# 添加节点
graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("reviewer", reviewer_node)
```

### 2.3 Edge（边）

```python
# 普通边：无条件转移
graph.add_edge("planner", "executor")
graph.add_edge("executor", "reviewer")

# 条件边：根据状态决定走向
def should_continue(state: AgentState) -> str:
    """条件路由函数"""
    last_message = state["messages"][-1].content

    if "已完成" in last_message or "最终答案" in last_message:
        return "end"
    elif state["step_count"] > 10:
        return "end"
    elif "需要更多信息" in last_message:
        return "executor"  # 继续执行
    else:
        return "planner"  # 重新规划

# 添加条件边
graph.add_conditional_edges(
    "reviewer",           # 从审查节点出发
    should_continue,      # 路由函数
    {
        "end": END,       # 结束
        "executor": "executor",  # 继续执行
        "planner": "planner",    # 重新规划
    }
)
```

### 2.4 Entry Point（入口）

```python
# 设置入口
graph.set_entry_point("planner")
# 或使用 START
graph.add_edge(START, "planner")
```

## 3. 完整示例：研究 Agent

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Literal
import operator

# 1. 定义状态
class ResearchState(TypedDict):
    question: str
    plan: str
    search_results: Annotated[list, operator.add]
    analysis: str
    answer: str
    iteration: int
    tools_used: Annotated[list, operator.add]

# 2. 定义节点
def planner(state: ResearchState) -> dict:
    """制定研究计划"""
    prompt = f"""
问题: {state['question']}
已有的搜索结果: {len(state['search_results'])} 条
当前迭代: {state['iteration']}

制定下一步计划。如果已有足够信息，输出 "READY_TO_ANSWER"。
"""
    plan = llm.invoke(prompt).content
    return {"plan": plan, "iteration": state["iteration"] + 1}

def searcher(state: ResearchState) -> dict:
    """执行搜索"""
    # 从计划中提取搜索查询
    query = extract_search_query(state["plan"])
    results = web_search(query)

    return {
        "search_results": [results],
        "tools_used": ["web_search"]
    }

def analyzer(state: ResearchState) -> dict:
    """分析搜索结果"""
    analysis = llm.invoke(f"""
问题: {state['question']}
搜索结果: {state['search_results']}

分析这些信息，找出关键要点。
""").content

    return {"analysis": analysis}

def answerer(state: ResearchState) -> dict:
    """生成最终答案"""
    answer = llm.invoke(f"""
基于以下研究结果回答问题：

问题: {state['question']}
分析: {state['analysis']}
搜索结果: {state['search_results']}

给出详细且有引用的答案。
""").content

    return {"answer": answer}

# 3. 路由函数
def route_after_planning(state: ResearchState) -> Literal["searcher", "answerer"]:
    if "READY_TO_ANSWER" in state["plan"]:
        return "answerer"
    return "searcher"

def route_after_analysis(state: ResearchState) -> Literal["planner", "answerer"]:
    if state["iteration"] >= 5:
        return "answerer"
    if "信息充分" in state["analysis"]:
        return "answerer"
    return "planner"

# 4. 构建图
workflow = StateGraph(ResearchState)

workflow.add_node("planner", planner)
workflow.add_node("searcher", searcher)
workflow.add_node("analyzer", analyzer)
workflow.add_node("answerer", answerer)

# 设置边
workflow.set_entry_point("planner")
workflow.add_conditional_edges("planner", route_after_planning)
workflow.add_edge("searcher", "analyzer")
workflow.add_conditional_edges("analyzer", route_after_analysis)
workflow.add_edge("answerer", END)

# 5. 编译和执行
app = workflow.compile()

result = app.invoke({
    "question": "2024年量子计算的重大突破有哪些？",
    "plan": "",
    "search_results": [],
    "analysis": "",
    "answer": "",
    "iteration": 0,
    "tools_used": []
})
```

## 4. 检查点与持久化 (Checkpointing)

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

# 内存检查点（开发用）
memory_checkpointer = MemorySaver()

# SQLite 检查点（持久化）
sqlite_checkpointer = SqliteSaver.from_conn_string("checkpoints.db")

# 编译时绑定检查点
app = workflow.compile(
    checkpointer=sqlite_checkpointer,
)

# 使用 thread_id 追踪会话
config = {"configurable": {"thread_id": "user-123-session-1"}}

# 第一次调用
result1 = app.invoke({"question": "什么是量子计算？"}, config)

# 恢复状态继续（会保留之前的状态）
result2 = app.invoke({"question": "它有什么应用？"}, config)

# 获取当前状态
state = app.get_state(config)
print(state.values)
```

## 5. 人工介入 (Human-in-the-Loop)

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

class ApprovalState(TypedDict):
    query: str
    plan: str
    approved: bool
    result: str

def generate_plan(state: ApprovalState) -> dict:
    plan = llm.invoke(f"为以下任务生成执行计划: {state['query']}").content
    return {"plan": plan}

def wait_for_approval(state: ApprovalState) -> dict:
    """暂停执行，等待人工批准"""
    print(f"计划:\n{state['plan']}")
    print("是否批准？(y/n)")
    # 实际场景中可能通过 API/webhook 获取审批
    return {}

def execute_plan(state: ApprovalState) -> dict:
    result = executor.run(state["plan"])
    return {"result": result}

# 构建图
workflow = StateGraph(ApprovalState)
workflow.add_node("plan", generate_plan)
workflow.add_node("wait", wait_for_approval)
workflow.add_node("execute", execute_plan)

workflow.set_entry_point("plan")
workflow.add_edge("plan", "wait")
workflow.add_conditional_edges("wait", lambda s: "execute" if s["approved"] else "plan")
workflow.add_edge("execute", END)

# 使用 interrupt_before 在特定节点前暂停
app = workflow.compile(
    interrupt_before=["wait"],  # 在 wait 节点前暂停
)
```

## 6. 多 Agent 编排

```python
# 多 Agent 系统的图结构
class MultiAgentState(TypedDict):
    task: str
    agent_outputs: Annotated[dict, lambda x, y: {**x, **y}]
    current_agent: str
    final_result: str

def supervisor(state: MultiAgentState) -> dict:
    """监督者：分配任务给合适的 Agent"""
    decision = llm.invoke(f"""
任务: {state['task']}
已有结果: {state['agent_outputs']}

选择下一个执行的 Agent: [researcher, coder, reviewer, COMPLETE]
""").content

    return {"current_agent": decision}

def researcher(state: MultiAgentState) -> dict:
    result = research_agent.invoke(state["task"])
    return {"agent_outputs": {"researcher": result}}

def coder(state: MultiAgentState) -> dict:
    result = coding_agent.invoke(state["task"])
    return {"agent_outputs": {"coder": result}}

def reviewer(state: MultiAgentState) -> dict:
    result = review_agent.invoke(state["agent_outputs"])
    return {"agent_outputs": {"reviewer": result}}

def route_supervisor(state: MultiAgentState) -> str:
    agent = state["current_agent"].strip()
    if agent == "COMPLETE":
        return "finalize"
    return agent

# 构建图
workflow = StateGraph(MultiAgentState)
workflow.add_node("supervisor", supervisor)
workflow.add_node("researcher", researcher)
workflow.add_node("coder", coder)
workflow.add_node("reviewer", reviewer)
workflow.add_node("finalize", lambda s: {"final_result": str(s["agent_outputs"])})

workflow.set_entry_point("supervisor")
workflow.add_conditional_edges("supervisor", route_supervisor)
for agent in ["researcher", "coder", "reviewer"]:
    workflow.add_edge(agent, "supervisor")
workflow.add_edge("finalize", END)

app = workflow.compile()
```

## 7. 可视化与调试

```python
# 生成图的可视化
from langgraph.graph import StateGraph

# 生成 Mermaid 图
print(app.get_graph().draw_mermaid())

# 输出示例:
# ```mermaid
# graph TD
#     START --> planner
#     planner --> searcher
#     planner --> answerer
#     searcher --> analyzer
#     analyzer --> planner
#     analyzer --> answerer
#     answerer --> END
# ```

# 生成 PNG 图片（需要 graphviz）
# app.get_graph().draw_png("agent_graph.png")

# 调试：打印每步状态
for event in app.stream({"question": "测试问题"}, stream_mode="updates"):
    print(f"节点: {list(event.keys())[0]}")
    print(f"状态更新: {event}")
    print("---")
```

## 总结

LangGraph 将 Agent 从**线性循环**提升为**图结构工作流**，核心能力包括：**(1) 状态管理** -- 完整的状态 Schema 和检查点；**(2) 条件路由** -- 动态决定执行路径；**(3) 人机协作** -- 关键节点支持人工介入。它是构建生产级复杂 Agent 的首选框架。
