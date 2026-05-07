# 9_AutoGPT 与自主 Agent

## 1. AutoGPT 概述

AutoGPT（2023.3）是首个引起广泛关注的**完全自主 Agent**，它能够自行设定子目标、执行操作并反思结果，无需人类逐步指导。

```
AutoGPT 核心循环:

用户目标 ─→ 设定子目标 ─→ 执行 ─→ 反思 ─→ 调整 ─→ 继续
                ↑                          │
                └──────────────────────────┘
                     持续自主循环

对比 ReAct:
- ReAct: 每步都需要用户提供问题，Agent 回答
- AutoGPT: 用户提供目标，Agent 自主规划并执行多步
```

## 2. 自主 Agent 架构

```python
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Goal:
    description: str
    priority: int = 1
    completed: bool = False
    sub_goals: list["Goal"] = field(default_factory=list)

@dataclass
class AgentMemory:
    short_term: list[str] = field(default_factory=list)  # 最近的思考和行动
    long_term: list[str] = field(default_factory=list)   # 持久化知识
    summary: str = ""  # 压缩后的历史

class AutonomousAgent:
    def __init__(self, llm, tools: dict, config: dict):
        self.llm = llm
        self.tools = tools
        self.goals: list[Goal] = []
        self.memory = AgentMemory()
        self.config = config
        self.step_count = 0
        self.max_steps = config.get("max_steps", 50)
        self.max_cycles = config.get("max_cycles", 10)

    def run(self, user_goal: str) -> str:
        """主循环"""
        # 1. 目标设定
        self.goals = self._decompose_goal(user_goal)

        for cycle in range(self.max_cycles):
            print(f"\n=== 循环 {cycle + 1} ===")

            # 2. 选择当前目标
            current_goal = self._select_next_goal()
            if not current_goal:
                return self._generate_final_report()

            # 3. 规划
            plan = self._plan(current_goal)
            print(f"[计划] {plan}")

            # 4. 执行
            result = self._execute(plan)
            print(f"[执行结果] {result[:200]}...")

            # 5. 反思
            reflection = self._reflect(current_goal, plan, result)
            print(f"[反思] {reflection}")

            # 6. 更新状态
            self._update_state(current_goal, reflection)

            # 7. 资源检查
            if self._check_resource_limits():
                break

        return self._generate_final_report()
```

## 3. 目标分解

```python
class GoalDecomposer:
    """将复杂目标分解为可执行的子目标"""

    def __init__(self, llm):
        self.llm = llm

    def decompose(self, main_goal: str) -> list[Goal]:
        prompt = f"""
将以下目标分解为具体的子目标列表。

主目标: {main_goal}

要求：
1. 子目标应按逻辑顺序排列
2. 每个子目标应可独立执行和验证
3. 标注优先级 (1-5，5最高)

输出 JSON:
[
  {{"description": "子目标描述", "priority": 5, "depends_on": []}},
  ...
]
"""
        response = self.llm.generate(prompt)
        sub_goals_data = json.loads(response)

        return [
            Goal(
                description=g["description"],
                priority=g.get("priority", 1)
            )
            for g in sub_goals_data
        ]

    def further_decompose(self, goal: Goal) -> list[Goal]:
        """递归分解：当子目标仍然太大时继续分解"""
        if self._is_atomic(goal):
            return [goal]

        sub_goals = self.decompose(goal.description)
        goal.sub_goals = sub_goals
        return sub_goals

    def _is_atomic(self, goal: Goal) -> bool:
        """判断目标是否已足够小（可单步执行）"""
        prompt = f"""
目标: {goal.description}

这个目标能否通过 1-2 个工具调用完成？
回答 "yes" 或 "no"。
"""
        return "yes" in self.llm.generate(prompt).lower()
```

## 4. 自我反思机制

```python
class SelfReflection:
    """Agent 自我反思模块"""

    def __init__(self, llm):
        self.llm = llm

    def reflect_on_action(self, goal: str, action: str,
                          result: str, history: list) -> dict:
        """反思单次行动"""
        prompt = f"""
反思以下行动的有效性：

目标: {goal}
行动: {action}
结果: {result[:500]}
历史: {history[-3:] if history else '无'}

请分析：
1. 行动是否推动了目标的完成？(score: 0-10)
2. 出现了什么问题？
3. 下一步应该怎么做？
4. 需要调整策略吗？

输出 JSON:
{{"score": X, "assessment": "评估", "next_action": "建议", "should_change_plan": bool}}
"""
        return json.loads(self.llm.generate(prompt))

    def reflect_on_progress(self, goals: list[Goal],
                            completed_actions: list) -> str:
        """全局反思：评估整体进展"""
        completed = [g for g in goals if g.completed]
        pending = [g for g in goals if not g.completed]

        prompt = f"""
整体进展评估：

已完成目标: {[g.description for g in completed]}
待完成目标: {[g.description for g in pending]}
已执行操作数: {len(completed_actions)}

请评估：
1. 整体完成进度 (0-100%)
2. 是否在正确的方向上？
3. 有没有更高效的方法？
4. 是否需要添加/删除/修改目标？
"""
        return self.llm.generate(prompt)
```

## 5. 资源管理

```python
class ResourceManager:
    """管理 Agent 的资源限制"""

    def __init__(self, config: dict):
        self.max_tokens = config.get("max_tokens", 100000)
        self.max_api_calls = config.get("max_api_calls", 50)
        self.max_time_seconds = config.get("max_time", 300)
        self.max_cost_usd = config.get("max_cost", 5.0)

        self.tokens_used = 0
        self.api_calls = 0
        self.start_time = time.time()
        self.cost_accumulated = 0.0

    def check_limits(self) -> tuple[bool, str]:
        """检查是否超出资源限制"""
        elapsed = time.time() - self.start_time

        if self.tokens_used >= self.max_tokens:
            return True, "Token 预算耗尽"
        if self.api_calls >= self.max_api_calls:
            return True, "API 调用次数耗尽"
        if elapsed >= self.max_time_seconds:
            return True, "执行时间超限"
        if self.cost_accumulated >= self.max_cost_usd:
            return True, "费用预算耗尽"

        return False, ""

    def record_usage(self, tokens: int, cost: float):
        self.tokens_used += tokens
        self.api_calls += 1
        self.cost_accumulated += cost

    def get_status(self) -> dict:
        return {
            "tokens_used": f"{self.tokens_used}/{self.max_tokens}",
            "api_calls": f"{self.api_calls}/{self.max_api_calls}",
            "time_elapsed": f"{time.time() - self.start_time:.1f}s",
            "cost": f"${self.cost_accumulated:.4f}/${self.max_cost_usd}",
        }
```

## 6. 防止失控的安全机制

```python
class SafetyController:
    """防止 Agent 行为失控"""

    def __init__(self, llm, config: dict):
        self.llm = llm
        self.forbidden_actions = config.get("forbidden", [
            "delete_all", "send_mass_email", "financial_transaction"
        ])
        self.action_log = []

    def check_action(self, proposed_action: str, args: dict) -> tuple[bool, str]:
        """检查拟执行的操作是否安全"""

        # 1. 黑名单检查
        action_name = args.get("tool_name", "")
        if action_name in self.forbidden_actions:
            return False, f"操作 '{action_name}' 在黑名单中"

        # 2. 重复操作检测
        action_sig = f"{action_name}:{json.dumps(args, sort_keys=True)}"
        recent = self.action_log[-5:]
        if recent.count(action_sig) >= 3:
            return False, "检测到重复操作，可能陷入循环"

        # 3. LLM 安全评估
        safety_check = self.llm.generate(f"""
评估以下操作是否存在风险：

操作: {proposed_action}
参数: {args}

风险等级: [LOW, MEDIUM, HIGH]
如果为 HIGH，说明原因并建议替代方案。
""")
        if "HIGH" in safety_check:
            return False, f"LLM 评估为高风险: {safety_check}"

        self.action_log.append(action_sig)
        return True, "安全"

    def enforce_output_limit(self, output: str, max_chars: int = 5000) -> str:
        if len(output) > max_chars:
            return output[:max_chars] + "\n... [已截断]"
        return output
```

## 7. 记忆与上下文管理

```python
class ContextManager:
    """管理 Agent 的上下文窗口"""

    def __init__(self, llm, max_context_tokens: int = 8000):
        self.llm = llm
        self.max_tokens = max_context_tokens
        self.history: list[dict] = []
        self.summary: str = ""

    def add(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        self._maybe_compress()

    def _maybe_compress(self):
        """当上下文接近上限时压缩"""
        estimated_tokens = sum(len(m["content"]) // 3 for m in self.history)

        if estimated_tokens > self.max_tokens * 0.8:
            self._compress()

    def _compress(self):
        """将旧历史压缩为摘要"""
        if len(self.history) <= 6:
            return

        old_history = self.history[:-4]  # 保留最近 4 条
        history_text = "\n".join(
            f"{m['role']}: {m['content'][:200]}" for m in old_history
        )

        self.summary = self.llm.generate(f"""
总结以下 Agent 执行历史的关键信息：
{history_text}

保留：关键决策、重要发现、错误教训、待完成事项。
""")

        self.history = [
            {"role": "system", "content": f"先前执行摘要:\n{self.summary}"}
        ] + self.history[-4:]

    def get_context(self) -> list[dict]:
        return self.history
```

## 8. AutoGPT 的局限与改进方向

```
AutoGPT 的已知问题：

1. 循环问题 (Looping)
   Agent 容易陷入无意义的重复操作
   → 解决：循环检测 + 惩罚机制 + 多样性约束

2. 目标漂移 (Goal Drift)
   Agent 偏离原始目标去做不相关的事
   → 解决：定期目标校验 + 子目标约束

3. 上下文丢失
   压缩历史时丢失关键信息
   → 解决：重要信息提取 + 向量记忆

4. 过度自信
   Agent 可能在不确定时给出错误答案
   → 解决：置信度评估 + 人工确认机制

5. 资源消耗
   多次 LLM 调用导致高成本和高延迟
   → 解决：小模型做简单决策 + 缓存 + 提前终止
```

## 9. 现代自主 Agent 框架对比

| 框架 | 特点 | 适用场景 |
|------|------|----------|
| **AutoGPT** | 完全自主，目标驱动 | 探索性任务 |
| **BabyAGI** | 任务队列驱动 | 研究类任务 |
| **MetaGPT** | 多角色协作 | 软件开发 |
| **CrewAI** | 角色+目标+工具 | 团队协作 |
| **AgentVerse** | 多 Agent 模拟 | 社会模拟 |

## 总结

AutoGPT 开创了"自主 Agent"范式，核心创新在于**目标驱动的自我规划和反思循环**。但其实践表明，完全自主仍面临循环、漂移和成本问题。后续框架（如 LangGraph）通过**更结构化的控制流**解决了部分问题，核心教训是：**自主性需要与可控性平衡**。
