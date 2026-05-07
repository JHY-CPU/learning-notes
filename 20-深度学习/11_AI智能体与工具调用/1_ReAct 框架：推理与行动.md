# 1_ReAct 框架：推理与行动

## 1. ReAct 核心思想

ReAct（Reasoning + Acting）由 Yao et al. (2022) 提出，核心思想是让 LLM **交替进行推理（Thought）和行动（Action）**，通过观察行动结果来调整后续推理。

```
传统方法对比：

Chain-of-Thought (CoT):  纯推理，无法与外部交互
      Thought → Thought → Thought → Answer

Act-Only:               纯行动，缺乏推理规划
      Action → Action → Action → Answer

ReAct:                   推理与行动交织
      Thought → Action → Observation → Thought → Action → ...
```

## 2. Thought-Action-Observation 循环

### 循环结构

```
┌─────────────────────────────────────────────┐
│              ReAct 循环                      │
│                                             │
│  ┌─────────┐                                │
│  │ Thought  │ ← LLM 分析当前状态，制定计划    │
│  └────┬────┘                                │
│       ↓                                     │
│  ┌─────────┐                                │
│  │ Action   │ ← 选择并执行具体工具/操作       │
│  └────┬────┘                                │
│       ↓                                     │
│  ┌─────────────┐                            │
│  │ Observation  │ ← 获取行动结果，更新状态     │
│  └────┬────────┘                            │
│       ↓                                     │
│  └──→ 循环直到任务完成                       │
└─────────────────────────────────────────────┘
```

### Prompt 模板

```python
REACT_SYSTEM_PROMPT = """你是一个能够使用工具的智能助手。

可用工具：
{tool_descriptions}

请按照以下格式执行任务：

Question: 用户的问题
Thought: 我需要思考如何解决这个问题
Action: 工具名称
Action Input: 工具参数
Observation: 工具返回结果
Thought: 基于结果的下一步思考
... (可以重复多次)
Final Answer: 最终答案

重要规则：
1. 每次行动后必须等待观察结果
2. 基于观察结果调整策略
3. 如果工具返回错误，尝试其他方法
"""
```

## 3. 完整实现

```python
from dataclasses import dataclass
from typing import Optional, Callable
import json
import re

@dataclass
class ToolResult:
    success: bool
    content: str

@dataclass
class AgentStep:
    thought: str
    action: Optional[str]
    action_input: Optional[str]
    observation: Optional[str]

class ReActAgent:
    def __init__(self, llm, tools: dict[str, Callable], max_steps: int = 8):
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.history: list[AgentStep] = []

    def run(self, question: str) -> str:
        prompt = self._build_prompt(question)

        for step_num in range(self.max_steps):
            # 1. 调用 LLM 获取下一步
            response = self.llm.generate(prompt)
            parsed = self._parse_response(response)

            step = AgentStep(
                thought=parsed["thought"],
                action=parsed.get("action"),
                action_input=parsed.get("action_input"),
                observation=None
            )

            # 2. 检查是否输出最终答案
            if parsed.get("final_answer"):
                return parsed["final_answer"]

            # 3. 执行工具调用
            if step.action and step.action in self.tools:
                try:
                    result = self.tools[step.action](step.action_input)
                    step.observation = str(result)
                except Exception as e:
                    step.observation = f"工具执行错误: {str(e)}"
            else:
                step.observation = f"未知工具: {step.action}"

            # 4. 更新 prompt 和历史
            self.history.append(step)
            prompt += self._step_to_string(step)

        return "达到最大步数限制，任务未能完成"

    def _parse_response(self, response: str) -> dict:
        result = {}

        thought_match = re.search(r"Thought:\s*(.+?)(?=\n|$)", response)
        if thought_match:
            result["thought"] = thought_match.group(1).strip()

        action_match = re.search(r"Action:\s*(\w+)", response)
        if action_match:
            result["action"] = action_match.group(1).strip()

        input_match = re.search(r"Action Input:\s*(.+?)(?=\n|$)", response)
        if input_match:
            result["action_input"] = input_match.group(1).strip()

        final_match = re.search(r"Final Answer:\s*(.+)", response, re.DOTALL)
        if final_match:
            result["final_answer"] = final_match.group(1).strip()

        return result

    def _build_prompt(self, question: str) -> str:
        tool_desc = "\n".join(
            f"- {name}: {func.__doc__ or '无描述'}"
            for name, func in self.tools.items()
        )
        return f"""可用工具:\n{tool_desc}\n\nQuestion: {question}\n"""

    def _step_to_string(self, step: AgentStep) -> str:
        s = f"Thought: {step.thought}\n"
        s += f"Action: {step.action}\n"
        s += f"Action Input: {step.action_input}\n"
        s += f"Observation: {step.observation}\n"
        return s
```

## 4. ReAct vs CoT 深度对比

| 维度 | Chain-of-Thought | ReAct |
|------|-----------------|-------|
| **外部交互** | 不支持 | 支持工具调用 |
| **信息来源** | 仅模型参数知识 | 参数知识 + 实时外部数据 |
| **幻觉风险** | 较高（无事实验证） | 较低（工具反馈纠正） |
| **推理深度** | 单次推理链 | 动态多步推理 |
| **适用场景** | 数学推理、逻辑题 | 事实查询、多步任务 |
| **延迟** | 较低 | 较高（多次 LLM 调用） |

### 幻觉抑制机制

```
CoT:  "巴黎人口约 1200 万"  ← 模型记忆，可能过时/错误
       (无验证机制)

ReAct: Thought: "我需要查巴黎的最新人口数据"
       Action: search("Paris population 2024")
       Observation: "巴黎市区人口约 210 万，都会区约 1200 万"
       Thought: "需要区分市区和都会区"
       Final Answer: "巴黎市区约 210 万，都会区约 1200 万"
       (通过工具验证，区分细节)
```

## 5. ReAct 的变体与改进

### 5.1 ReAct + CoT 混合

```python
HYBRID_PROMPT = """任务处理策略：
1. 简单推理题 → 直接用 CoT 推理，无需工具
2. 事实查询题 → 使用 ReAct 调用搜索工具
3. 复杂任务 → CoT 分解子任务，ReAct 逐个执行

示例：
Question: 计算 (23 + 45) * 3
Thought: 这是纯数学计算，不需要外部工具
Final Answer: (23 + 45) * 3 = 68 * 3 = 204

Question: 2024年诺贝尔物理学奖得主是谁？
Thought: 这需要查询最新信息，使用搜索工具
Action: search("2024 Nobel Prize Physics winner")
...
"""
```

### 5.2 带反思的 ReAct (Reflexion)

```python
class ReflexionAgent(ReActAgent):
    def run(self, question: str) -> str:
        result = super().run(question)

        # 自我反思
        reflection = self.llm.generate(f"""
任务: {question}
执行过程: {self._format_history()}
最终结果: {result}

请反思：
1. 过程中是否有错误决策？
2. 哪些步骤可以优化？
3. 如果重做会怎么改进？
""")
        self.reflections.append(reflection)
        return result
```

## 6. 常见问题与调试

### 6.1 Action 解析失败

```python
def robust_parse(response: str) -> dict:
    """增强版解析，处理格式不规范的输出"""
    result = {}

    # 尝试多种格式
    patterns = {
        "thought": [r"Thought:\s*(.+)", r"思考[：:]\s*(.+)"],
        "action": [r"Action:\s*(\w+)", r"行动[：:]\s*(\w+)"],
        "action_input": [r"Action Input:\s*(.+)", r"输入[：:]\s*(.+)"],
    }

    for key, pats in patterns.items():
        for pat in pats:
            match = re.search(pat, response)
            if match:
                result[key] = match.group(1).strip()
                break

    return result
```

### 6.2 死循环检测

```python
class SafeReActAgent(ReActAgent):
    def run(self, question: str) -> str:
        action_history = []

        for step_num in range(self.max_steps):
            response = self.llm.generate(self.prompt)
            parsed = self._parse_response(response)

            # 死循环检测：连续相同动作
            if parsed.get("action"):
                action_signature = (parsed["action"], parsed.get("action_input"))
                if action_signature in action_history[-3:]:
                    self.prompt += "\nThought: 我重复了相同的动作，需要换一种方法\n"
                    continue
                action_history.append(action_signature)

            # ... 正常处理流程
```

## 7. 性能优化实践

```python
# 优化 1: 缓存工具结果
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_search(query: str) -> str:
    return search_api(query)

# 优化 2: 并行独立工具调用
import asyncio

async def parallel_tool_calls(tools_to_call: list[tuple[str, str]]):
    tasks = [
        execute_tool_async(name, args)
        for name, args in tools_to_call
    ]
    return await asyncio.gather(*tasks)

# 优化 3: 早期停止 — 置信度足够高时直接输出
def should_stop_early(self, thought: str) -> bool:
    confidence_signals = [
        "我已经有了完整答案",
        "信息已经足够",
        "可以确定" in thought,
    ]
    return any(confidence_signals)
```

## 8. 实战示例：多步研究 Agent

```python
def create_research_agent():
    tools = {
        "search": web_search,
        "read_url": fetch_page_content,
        "calculate": safe_calculate,
        "final_answer": lambda x: x,
    }

    agent = ReActAgent(
        llm=gpt4,
        tools=tools,
        max_steps=10
    )

    # 执行任务
    result = agent.run("比较 2024 年 GPT-4 和 Claude 3 的性能差异")
    return result

# 预期执行流程：
# Thought: 需要搜索 GPT-4 和 Claude 3 的基准测试结果
# Action: search
# Action Input: "GPT-4 vs Claude 3 benchmark comparison 2024"
# Observation: [搜索结果...]
# Thought: 需要查看具体的评测数据
# Action: read_url
# Action Input: "https://example.com/benchmark-results"
# ...
```

## 总结

ReAct 是 Agent 架构的基石范式。其核心价值在于：**(1) 推理为行动提供方向，行动为推理提供事实依据；(2) 通过工具反馈抑制幻觉；(3) 支持多步复杂任务分解**。掌握 ReAct 是理解后续更复杂 Agent 框架的前提。
