# 0_AI Agent 概述与范式演进

## 1. 什么是 AI Agent？

AI Agent（智能体）是一种能够**自主感知环境、做出决策并采取行动**以达成目标的人工智能系统。与传统的"输入-输出"模型不同，Agent 具备**循环推理**和**工具使用**能力，能够在复杂环境中持续交互。

```
┌─────────────────────────────────────────────────┐
│                  AI Agent 核心循环                │
│                                                  │
│   感知 (Perceive) → 规划 (Plan) → 行动 (Act)     │
│        ↑                              │          │
│        └──────── 反馈 (Observe) ←─────┘          │
│                                                  │
│   + 记忆 (Memory) 贯穿整个循环                    │
└─────────────────────────────────────────────────┘
```

### Agent 的四大核心组成

| 组件 | 功能 | 典型实现 |
|------|------|----------|
| **感知 (Perception)** | 接收用户输入、环境信号 | LLM 文本理解、多模态输入 |
| **规划 (Planning)** | 分解目标、制定执行计划 | CoT、ToT、任务分解 |
| **记忆 (Memory)** | 存储和检索历史信息 | 上下文窗口、向量数据库 |
| **行动 (Action)** | 执行具体操作 | API 调用、代码执行、工具使用 |

## 2. 范式演进：从规则系统到 LLM Agent

### 第一代：规则驱动的专家系统 (1960s-1990s)

```python
# 专家系统示例：基于规则的诊断系统
class ExpertSystem:
    def __init__(self):
        self.rules = [
            {"if": ["发烧", "咳嗽"], "then": "可能是感冒"},
            {"if": ["发烧", "皮疹"], "then": "可能是麻疹"},
            {"if": ["头痛", "恶心"], "then": "可能是偏头痛"},
        ]

    def diagnose(self, symptoms: list[str]) -> str:
        for rule in self.rules:
            if all(s in symptoms for s in rule["if"]):
                return rule["then"]
        return "无法诊断"
```

**特点**：规则完全由人类编写，无法处理未见过的情况，扩展性差。

### 第二代：统计机器学习 Agent (2000s-2010s)

```python
# 基于强化学习的简单 Agent
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99):
        self.q_table = np.zeros((n_states, n_actions))
        self.lr = lr
        self.gamma = gamma

    def choose_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(self.q_table.shape[1])
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state):
        predict = self.q_table[state, action]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state, action] += self.lr * (target - predict)
```

**特点**：能从数据中学习策略，但泛化能力有限，难以处理自然语言任务。

### 第三代：LLM 驱动的 Agent (2022-present)

```python
# 基于 LLM 的现代 Agent 框架
from typing import Callable

class LLMAgent:
    def __init__(self, llm, tools: dict[str, Callable], system_prompt: str):
        self.llm = llm
        self.tools = tools
        self.messages = [{"role": "system", "content": system_prompt}]
        self.max_steps = 10

    def run(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        for step in range(self.max_steps):
            # LLM 决定下一步行动
            response = self.llm.chat(self.messages, tools=self.tools)

            if response.type == "final_answer":
                return response.content

            # 执行工具调用
            if response.type == "tool_call":
                result = self.execute_tool(response.tool_name, response.arguments)
                self.messages.append({
                    "role": "tool",
                    "content": str(result),
                    "tool_call_id": response.id
                })

        return "达到最大步数限制"

    def execute_tool(self, name: str, args: dict):
        return self.tools[name](**args)
```

**特点**：具备通用推理能力，可通过工具扩展能力边界，能处理开放域任务。

## 3. Agent 范式分类

### 3.1 按自主程度分类

```
自主程度从低到高：

Reactive Agent ──→ Deliberative Agent ──→ Autonomous Agent
   (反应式)           (深思式)              (自主式)

- 被动响应输入        - 主动规划步骤        - 自主设定目标
- 单步操作            - 多步推理链          - 持续自我改进
- 无记忆              - 有工作记忆          - 长期记忆+反思
```

### 3.2 按架构分类

| 类型 | 描述 | 代表工作 |
|------|------|----------|
| **单 Agent** | 单个 LLM 驱动，自主完成任务 | ReAct, AutoGPT |
| **Multi-Agent** | 多个 Agent 协作/竞争 | CrewAI, AutoGen, MetaGPT |
| **层级 Agent** | 管理者 Agent 调度子 Agent | Hierarchical Agent |
| **混合 Agent** | LLM + 传统 ML 模型组合 | Toolformer, HuggingGPT |

## 4. Agent vs 传统 Chatbot

```
┌──────────────────┬────────────────────┬────────────────────┐
│     维度          │    传统 Chatbot    │     AI Agent       │
├──────────────────┼────────────────────┼────────────────────┤
│  交互模式         │  被动问答          │  主动规划+执行      │
│  工具使用         │  无/有限           │  动态调用外部工具   │
│  任务复杂度       │  单轮/多轮对话      │  多步骤复杂任务     │
│  状态管理         │  短期上下文        │  长期记忆+状态追踪   │
│  错误处理         │  静态回复          │  自我反思+重试      │
│  典型场景         │  FAQ、闲聊         │  代码生成、数据分析  │
└──────────────────┴────────────────────┴────────────────────┘
```

## 5. Agent 技术栈全景

```
┌─────────────────────────────────────────────────────────┐
│                     应用层 (Applications)                │
│   Code Agent │ Web Agent │ 数据分析 Agent │ 客服 Agent   │
├─────────────────────────────────────────────────────────┤
│                     编排层 (Orchestration)               │
│   LangChain │ LangGraph │ CrewAI │ AutoGen             │
├─────────────────────────────────────────────────────────┤
│                     能力层 (Capabilities)                │
│   ReAct │ CoT/ToT │ Function Calling │ RAG │ Memory    │
├─────────────────────────────────────────────────────────┤
│                     模型层 (Foundation Models)           │
│   GPT-4 │ Claude │ Llama │ Gemini │ Qwen              │
├─────────────────────────────────────────────────────────┤
│                     基础设施层 (Infrastructure)          │
│   向量数据库 │ 沙箱执行 │ 监控日志 │ API 网关           │
└─────────────────────────────────────────────────────────┘
```

## 6. 关键里程碑

| 时间 | 事件 | 意义 |
|------|------|------|
| 2022.10 | ReAct 论文发布 | 提出 Thought-Action-Observation 循环 |
| 2023.03 | AutoGPT 开源 | 首个引起广泛关注的自主 Agent |
| 2023.06 | OpenAI Function Calling | 标准化工具调用接口 |
| 2023.10 | LangGraph 发布 | 有状态 Agent 编排框架 |
| 2024.01 | MCP 协议提出 | 工具标准化协议 |
| 2024.06 | Claude Artifacts | Agent 与代码执行结合 |
| 2025.01 | SWE-bench 普及 | Agent 代码能力基准测试 |

## 7. 本章学习路径

```
基础概念 (0-2) → 工具调用 (3-4) → 记忆与检索 (5-6)
      ↓              ↓                ↓
框架实战 (7-8) → 特定领域 Agent (9-12) → 评估与安全 (13-14)
      ↓              ↓                ↓
企业应用 (15-16) → 工程实践 (17-18) → 前沿方向 (19)
```

## 8. 核心概念速查

- **Agent Loop**: 感知→规划→行动→观察 的循环执行
- **Tool Use**: Agent 调用外部 API/函数 扩展能力
- **Grounding**: 将 LLM 输出与真实世界操作连接
- **Hallucination Control**: 通过工具反馈减少幻觉
- **Scaffolding**: 为 LLM 提供结构化的执行框架

## 总结

AI Agent 代表了从"语言模型"到"行动模型"的关键转变。核心洞察是：**LLM 作为"大脑"负责推理和决策，工具作为"四肢"负责与外部世界交互，记忆作为"海马体"负责信息存储与检索**。理解这个架构是后续深入学习各类 Agent 技术的基础。
