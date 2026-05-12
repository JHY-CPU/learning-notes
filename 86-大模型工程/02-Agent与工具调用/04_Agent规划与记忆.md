# Agent规划与记忆 - Agent与工具调用

*ReAct、Reflexion、ToT/GoT 规划策略，短期/长期/情景记忆与自我反思机制*

规划策略深度对比

| 策略 | 核心思想 | 探索方式 | 适用场景 | 代表论文 |
| --- | --- | --- | --- | --- |
| **ReAct** | 交替思考与行动 | 单路径串行 | 通用 Agent | Yao et al. 2022 |
| **Reflexion** | 失败后反思改进 | 试错迭代 | 需要学习的任务 | Shinn et al. 2023 |
| **CoT** | 逐步推理 | 线性推理链 | 逻辑推理 | Wei et al. 2022 |
| **ToT** | 树形搜索多路径 | BFS/DFS | 需要探索的问题 | Yao et al. 2023 |
| **GoT** | 图结构推理 | DAG 聚合 | 复杂多步推理 | Besta et al. 2023 |
| **LATS** | 语言 Agent 树搜索 | MCTS | 需要长期规划 | Zhou et al. 2023 |

ToT vs GoT 核心差异

| 维度 | ToT（思维树） | GoT（思维图） |
| --- | --- | --- |
| 结构 | 树（Tree） | 有向无环图（DAG） |
| 合并 | 不支持 | 多条路径可合并为一个节点 |
| 回溯 | DFS 回溯 | 任意路径回环 |
| 复杂度 | 中 | 高 |
| 适用 | 探索性问题 | 需要多路径聚合的复杂推理 |

记忆体系完整架构

| 记忆类型 | 认知科学类比 | 存储介质 | 生命周期 | 实现方式 |
| --- | --- | --- | --- | --- |
| **工作记忆** | 工作记忆 | LLM 上下文窗口 | 单次推理 | Prompt 中的系统指令 + 当前状态 |
| **短期记忆** | 短期记忆 | 对话缓冲区 | 当前会话 | 消息列表 + 滑动窗口 |
| **长期记忆** | 长期记忆 | 外部存储 | 持久化 | 向量数据库 / KV 存储 |
| **情景记忆** | 事件记忆 | 事件日志 | 持久化 | 任务执行记录 + 检索 |
| **语义记忆** | 事实记忆 | 知识库 | 持久化 | 知识图谱 / 结构化存储 |
| **程序记忆** | 技能记忆 | 技能库 | 持久化 | 成功案例模板 + 工具使用记录 |

## ReAct 框架详解

ReAct 交替执行推理(Reasoning)和行动(Acting)，是最主流的 Agent 框架。

```python
# ReAct 循环伪代码
def react_agent(query, tools, max_steps=10):
    context = query
    for step in range(max_steps):
        # 1. 思考
        thought = llm(f"思考: 基于当前信息，下一步应该...")

        # 2. 行动
        if thought.should_act:
            action = thought.action  # 如 Search("xxx")
            result = tools.execute(action)
            context += f"\n观察: {result}"
        else:
            # 3. 得出最终答案
            return llm(f"基于以下信息回答问题:\n{context}")
```

ReAct 的 Prompt 模板：
```
问题: {question}

你可以使用以下工具:
{tool_descriptions}

请按以下格式:
思考: 你的推理过程
行动: 工具名称[参数]
观察: 工具返回结果
... (重复 N 次)
最终答案: 你的回答
```

## Reflexion 自我反思

```python
# Reflexion 流程
def reflexion_agent(task, max_trials=3):
    for trial in range(max_trials):
        # 执行任务
        result = execute(task)

        # 评估结果
        if evaluate(result, task):
            return result

        # 失败后反思
        reflection = llm(f"""
        任务: {task}
        尝试: {result}
        评价: {evaluate_detail(result)}

        反思: 哪里出了问题？下次如何改进？
        """)
        # 将反思加入下次尝试的上下文
        task = f"{task}\n\n之前的反思:\n{reflection}"

    return result
```

## 记忆系统的实现

### 短期记忆（对话历史）

```python
# 滑动窗口方式
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(
    k=10,  # 保留最近10轮对话
    return_messages=True,
)

# Token 限制方式
from langchain.memory import ConversationTokenBufferMemory
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000,
)
```

### 长期记忆（向量数据库）

```python
# 使用向量数据库存储长期记忆
from langchain.memory import VectorStoreRetrieverMemory
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_texts([], embedding)
memory = VectorStoreRetrieverMemory(
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
)

# 存储记忆
memory.save_context(
    {"input": "我喜欢编程"},
    {"output": "好的，我记住了你对编程的兴趣"}
)

# 检索相关记忆
relevant = memory.load_memory_variables({"input": "我的爱好是什么"})
```

### 情景记忆（任务日志）

```python
# 记录每次任务执行的详细信息
task_log = {
    "task_id": "task_001",
    "goal": "订一张去北京的机票",
    "steps_taken": [
        {"action": "search_flights", "result": "找到3个航班"},
        {"action": "select_flight", "result": "选择了CA1234"},
    ],
    "outcome": "success",
    "reflection": "下次应该先确认日期再搜索",
    "timestamp": "2024-01-15T10:30:00Z",
}
```

## Agent 常见陷阱

1. **无限循环**：Agent 反复执行相同操作，需要设置最大步数
2. **幻觉工具调用**：Agent 调用不存在的工具，需要严格的工具验证
3. **上下文窗口溢出**：长期运行导致上下文过长，需要记忆压缩
4. **安全风险**：Agent 执行危险操作，需要权限控制和人工确认机制
5. **成本失控**：多次 LLM 调用累积成本，需要预算限制和路由优化


<!-- Converted from: 04_Agent规划与记忆.html -->
