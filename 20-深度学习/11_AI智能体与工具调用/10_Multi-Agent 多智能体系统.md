# 10_Multi-Agent 多智能体系统

## 1. 多智能体系统概述

多智能体（Multi-Agent）系统由**多个独立的 LLM Agent 协同工作**，每个 Agent 有独立的角色、目标和工具，通过协作或竞争完成复杂任务。

```
单 Agent 局限性：
- 上下文窗口有限，难以处理超长任务
- 单一视角，缺乏多样性
- 难以并行执行多个子任务
- 能力受限于单个模型的知识范围

Multi-Agent 优势：
- 专业化分工，每个 Agent 专精一个领域
- 并行执行，提高效率
- 多视角辩论，提高决策质量
- 模块化扩展，便于维护
```

## 2. 协作模式

### 2.1 层级模式 (Hierarchical)

```
         Manager Agent (任务分配与监督)
        /        |         \
    Worker₁   Worker₂   Worker₃
    (研究)     (编码)    (审查)
        \        |         /
         Manager (汇总结果)
```

```python
class ManagerAgent:
    def __init__(self, llm, workers: dict[str, Agent]):
        self.llm = llm
        self.workers = workers

    def execute(self, task: str) -> str:
        # 分解任务
        subtasks = self.decompose(task)

        results = {}
        for subtask in subtasks:
            # 选择合适的 Worker
            worker_name = self.assign_worker(subtask)
            result = self.workers[worker_name].execute(subtask["description"])
            results[subtask["id"]] = result

            # 检查质量
            quality = self.review(result, subtask["criteria"])
            if quality < 0.7:
                # 重新分配或让其他 Worker 复查
                result = self.workers["reviewer"].review_and_fix(result)
                results[subtask["id"]] = result

        return self.synthesize(results)
```

### 2.2 平等协作模式 (Flat)

```
    Agent₁ ←→ Agent₂
      ↕    ╲ ╱    ↕
    Agent₃ ←→ Agent₄

每个 Agent 地位平等，通过消息传递协作
```

```python
class FlatMultiAgent:
    def __init__(self, agents: list[Agent]):
        self.agents = agents
        self.message_queue = []

    def collaborate(self, task: str) -> str:
        # 广播任务
        for agent in self.agents:
            response = agent.process(task)
            self.broadcast(agent.name, response)

        # 迭代讨论
        for round_num in range(5):
            for agent in self.agents:
                messages = self.get_messages_for(agent)
                response = agent.discuss(messages)
                self.broadcast(agent.name, response)

                # 检查共识
                if self.check_consensus():
                    return self.get_consensus_result()

        return self.majority_vote()
```

### 2.3 辩论模式 (Debate)

```python
class DebateAgents:
    def __init__(self, llm):
        self.proponent = self.create_agent("支持者", "从正面角度论证")
        self.opponent = self.create_agent("反对者", "从反面角度质疑")
        self.judge = self.create_agent("裁判", "综合评估双方观点")

    def debate(self, topic: str, rounds: int = 3) -> str:
        arguments = {"pro": [], "con": []}

        for r in range(rounds):
            # 支持方发言
            pro_arg = self.proponent.argue(
                topic, arguments["con"][-1] if arguments["con"] else None
            )
            arguments["pro"].append(pro_arg)

            # 反对方发言
            con_arg = self.opponent.argue(
                topic, arguments["pro"][-1]
            )
            arguments["con"].append(con_arg)

        # 裁判裁决
        verdict = self.judge.evaluate(topic, arguments)
        return verdict
```

## 3. CrewAI 框架

```python
from crewai import Agent, Task, Crew, Process

# 定义 Agent
researcher = Agent(
    role="资深研究员",
    goal="深入研究给定主题，收集全面准确的信息",
    backstory="""你是一位经验丰富的研究员，擅长从多个角度
    分析问题，善于发现数据中的规律和趋势。""",
    tools=[web_search, read_pdf, calculator],
    llm="gpt-4",
    verbose=True
)

writer = Agent(
    role="技术作家",
    goal="将研究成果转化为清晰、有逻辑的文章",
    backstory="""你是一位优秀的技术作家，擅长将复杂的概念
    用简洁的语言表达，文章结构严谨，逻辑清晰。""",
    llm="gpt-4",
    verbose=True
)

reviewer = Agent(
    role="质量审查员",
    goal="审查文章的准确性、完整性和可读性",
    backstory="""你是一位严谨的审查员，注重细节，
    善于发现文章中的逻辑漏洞和事实错误。""",
    llm="gpt-4",
    verbose=True
)

# 定义任务
research_task = Task(
    description="研究 {topic} 的最新进展和关键技术",
    expected_output="包含关键发现、数据支撑和参考来源的研究报告",
    agent=researcher
)

writing_task = Task(
    description="基于研究报告，撰写一篇面向技术读者的文章",
    expected_output="结构完整、论据充分的技术文章",
    agent=writer,
    context=[research_task]  # 依赖研究任务的输出
)

review_task = Task(
    description="审查文章的准确性、逻辑性和可读性",
    expected_output="审查报告和改进建议",
    agent=reviewer,
    context=[writing_task]
)

# 组建团队
crew = Crew(
    agents=[researcher, writer, reviewer],
    tasks=[research_task, writing_task, review_task],
    process=Process.sequential,  # 顺序执行
    # process=Process.hierarchical,  # 层级执行
    verbose=True
)

# 执行
result = crew.kickoff(inputs={"topic": "2024年大语言模型推理优化技术"})
```

## 4. AutoGen 框架

```python
import autogen

# 配置 LLM
config_list = [
    {"model": "gpt-4", "api_key": "your-api-key"}
]

# 创建 Agent
assistant = autogen.AssistantAgent(
    name="Assistant",
    llm_config={"config_list": config_list},
    system_message="你是一个专业的编程助手，精通 Python 和数据科学。"
)

coder = autogen.AssistantAgent(
    name="Coder",
    llm_config={"config_list": config_list},
    system_message="你是一个代码专家。编写高质量、有文档的代码。"
)

user_proxy = autogen.UserProxyAgent(
    name="UserProxy",
    human_input_mode="TERMINATE",  # 完成后请求用户确认
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "workspace",
        "use_docker": False,
    }
)

# 群组对话
groupchat = autogen.GroupChat(
    agents=[user_proxy, assistant, coder],
    messages=[],
    max_round=20
)

manager = autogen.GroupChatManager(groupchat=groupchat)

# 启动对话
user_proxy.initiate_chat(
    manager,
    message="帮我分析 sales_data.csv 文件，找出销售趋势并生成可视化图表。"
)
```

### AutoGen 的对话模式

```python
# 两 Agent 对话
user_proxy.initiate_chat(
    assistant,
    message="解释 Transformer 的自注意力机制"
)

# 带代码执行的对话
user_proxy.initiate_chat(
    assistant,
    message="写一个计算 Fibonacci 数列第 n 项的函数，并测试"
)

# 嵌套对话
def nested_chat(task):
    # 内部对话
    inner_result = coder.initiate_chat(
        assistant,
        message=f"实现: {task}"
    )
    # 外部对话使用内部结果
    return reviewer.initiate_chat(
        assistant,
        message=f"审查代码: {inner_result}"
    )
```

## 5. 通信协议

```python
from dataclasses import dataclass
from enum import Enum
from typing import Any

class MessageType(Enum):
    TASK_ASSIGN = "task_assign"
    RESULT_REPORT = "result_report"
    QUESTION = "question"
    ANSWER = "answer"
    BROADCAST = "broadcast"
    NEGOTIATION = "negotiation"

@dataclass
class AgentMessage:
    sender: str
    receiver: str
    msg_type: MessageType
    content: Any
    timestamp: float
    reply_to: str | None = None
    priority: int = 0

class MessageBus:
    """Agent 间通信的消息总线"""

    def __init__(self):
        self.subscribers: dict[str, list[Agent]] = {}
        self.message_log: list[AgentMessage] = []

    def publish(self, message: AgentMessage):
        self.message_log.append(message)

        if message.receiver == "broadcast":
            for agents in self.subscribers.values():
                for agent in agents:
                    agent.receive(message)
        elif message.receiver in self.subscribers:
            for agent in self.subscribers[message.receiver]:
                agent.receive(message)

    def subscribe(self, topic: str, agent: Agent):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(agent)
```

## 6. 冲突解决

```python
class ConflictResolver:
    """解决 Agent 间的决策冲突"""

    def resolve(self, proposals: list[dict], method: str = "vote") -> dict:
        if method == "vote":
            return self.majority_vote(proposals)
        elif method == "weighted_vote":
            return self.weighted_vote(proposals)
        elif method == "debate":
            return self.debated_resolution(proposals)
        elif method == "supervisor":
            return self.supervisor_decision(proposals)

    def majority_vote(self, proposals: list[dict]) -> dict:
        from collections import Counter
        votes = Counter(p["choice"] for p in proposals)
        winner = votes.most_common(1)[0][0]
        return {"decision": winner, "method": "majority_vote", "votes": dict(votes)}

    def weighted_vote(self, proposals: list[dict]) -> dict:
        scores = {}
        for p in proposals:
            choice = p["choice"]
            weight = p.get("confidence", 0.5) * p.get("expertise", 1.0)
            scores[choice] = scores.get(choice, 0) + weight

        winner = max(scores, key=scores.get)
        return {"decision": winner, "method": "weighted_vote", "scores": scores}
```

## 7. 多 Agent 系统设计模式

```
┌─────────────────────────────────────────────────────────┐
│               Multi-Agent 设计模式                       │
├──────────┬──────────────────────────────────────────────┤
│ 监督者    │ 一个 Agent 管理和协调其他 Agent              │
│ 对等协作  │ Agent 平等协商，投票决策                     │
│ 竞争/辩论 │ 多个 Agent 从不同角度辩论，提升决策质量       │
│ 流水线    │ Agent 按顺序处理，前一个输出是后一个输入       │
│ 市场      │ Agent 通过竞标/拍卖分配任务                  │
│ 细胞自动机 │ 简单规则+大量 Agent 涌现复杂行为             │
└──────────┴──────────────────────────────────────────────┘
```

## 总结

Multi-Agent 系统通过**专业化分工和多视角协作**突破了单 Agent 的能力边界。核心挑战在于**通信效率、冲突解决和成本控制**。CrewAI 和 AutoGen 是目前最流行的两大框架，分别侧重团队协作和对话式协作。选择取决于任务特点：顺序任务用 CrewAI，开放式讨论用 AutoGen。
