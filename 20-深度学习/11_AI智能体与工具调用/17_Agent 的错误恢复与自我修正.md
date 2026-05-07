# 17_Agent 的错误恢复与自我修正

## 1. Agent 错误类型分类

```
Agent 错误分类树：

Agent 错误
├── 感知错误
│   ├── 输入理解错误（误解用户意图）
│   └── 观察解析错误（错误解读工具返回值）
├── 规划错误
│   ├── 目标设定错误（偏离用户需求）
│   ├── 步骤规划错误（逻辑顺序不当）
│   └── 工具选择错误（选了不合适的工具）
├── 执行错误
│   ├── 参数错误（工具参数格式/值错误）
│   ├── 工具故障（工具自身执行失败）
│   └── 环境错误（网络超时、权限不足）
└── 输出错误
    ├── 事实错误（幻觉/过时信息）
    └── 格式错误（输出格式不符合要求）
```

## 2. 失败检测机制

```python
class FailureDetector:
    """Agent 执行失败检测"""

    def detect(self, step: dict, result: dict,
               context: dict) -> dict:
        """检测当前步骤是否失败"""
        failures = []

        # 1. 显式错误检测
        if not result.get("success", True):
            failures.append({
                "type": "execution_error",
                "detail": result.get("error", "未知错误"),
                "severity": "high"
            })

        # 2. 循环检测
        if self.detect_loop(context.get("history", [])):
            failures.append({
                "type": "loop_detected",
                "detail": "Agent 陷入重复操作循环",
                "severity": "high"
            })

        # 3. 目标偏离检测
        if self.detect_drift(step, context):
            failures.append({
                "type": "goal_drift",
                "detail": "当前操作偏离了原始目标",
                "severity": "medium"
            })

        # 4. 幻觉检测
        if self.detect_hallucination(result, context):
            failures.append({
                "type": "hallucination",
                "detail": "输出内容可能包含不实信息",
                "severity": "medium"
            })

        return {
            "has_failure": len(failures) > 0,
            "failures": failures,
            "action": self.decide_action(failures)
        }

    def detect_loop(self, history: list, window: int = 5) -> bool:
        """检测重复循环"""
        if len(history) < window:
            return False

        recent = history[-window:]
        signatures = [
            f"{h.get('tool', '')}:{json.dumps(h.get('args', {}), sort_keys=True)}"
            for h in recent
        ]

        # 如果窗口内有超过 60% 的操作相同，判定为循环
        from collections import Counter
        most_common_count = Counter(signatures).most_common(1)[0][1]
        return most_common_count / window > 0.6

    def detect_drift(self, current_step: dict, context: dict) -> bool:
        """检测目标漂移"""
        goal = context.get("original_goal", "")
        current_action = current_step.get("description", "")

        # 简单的关键词匹配（生产环境用语义相似度）
        goal_keywords = set(goal.lower().split())
        action_keywords = set(current_action.lower().split())
        overlap = len(goal_keywords & action_keywords) / max(len(goal_keywords), 1)

        return overlap < 0.2

    def decide_action(self, failures: list) -> str:
        """根据失败类型决定恢复策略"""
        if not failures:
            return "continue"

        severities = [f["severity"] for f in failures]

        if "high" in severities:
            return "retry_with_different_approach"
        elif "medium" in severities:
            return "self_correct"
        else:
            return "continue_with_warning"
```

## 3. 重试策略

```python
class RetryStrategy:
    """智能重试策略"""

    def __init__(self, llm, max_retries: int = 3):
        self.llm = llm
        self.max_retries = max_retries
        self.retry_history = []

    def retry(self, original_plan: str, error: dict,
              attempt: int) -> dict:
        """生成修正后的执行计划"""

        # 策略1: 简单重试（适用于临时错误）
        if error["type"] == "network_timeout" and attempt == 1:
            return {"strategy": "simple_retry", "plan": original_plan}

        # 策略2: 参数修正
        if error["type"] == "parameter_error":
            return self.fix_parameters(original_plan, error)

        # 策略3: 工具替换
        if error["type"] == "tool_failure" and attempt >= 2:
            return self.find_alternative_tool(original_plan, error)

        # 策略4: LLM 重新规划
        return self.replan(original_plan, error, attempt)

    def fix_parameters(self, plan: str, error: dict) -> dict:
        """修正工具参数"""
        correction = self.llm.generate(f"""
原始计划: {plan}
错误信息: {error['detail']}

分析错误原因，修正参数：
输出 JSON: {{"corrected_params": {{...}}, "reason": "..."}}
""")
        return {"strategy": "parameter_fix", "correction": json.loads(correction)}

    def find_alternative_tool(self, plan: str, error: dict) -> dict:
        """寻找替代工具"""
        alternative = self.llm.generate(f"""
工具调用失败。原始计划: {plan}
失败工具: {error.get('tool_name')}
错误: {error['detail']}

推荐替代工具和修改后的计划。
""")
        return {"strategy": "tool_alternative", "new_plan": alternative}

    def replan(self, original_plan: str, error: dict, attempt: int) -> dict:
        """完全重新规划"""
        new_plan = self.llm.generate(f"""
任务执行失败，需要重新规划。

原始计划: {plan}
已尝试 {attempt} 次，最后一次错误: {error['detail']}

请生成一个完全不同的执行策略。
注意：避免重复之前失败的方法。
""")
        return {"strategy": "full_replan", "plan": new_plan}
```

## 4. 自我修正框架

```python
class SelfCorrectingAgent:
    """带自我修正能力的 Agent"""

    def __init__(self, llm, tools: dict, config: dict):
        self.llm = llm
        self.tools = tools
        self.detector = FailureDetector()
        self.retry = RetryStrategy(llm, config.get("max_retries", 3))
        self.max_corrections = config.get("max_corrections", 3)

    def run(self, task: str) -> dict:
        context = {
            "original_goal": task,
            "history": [],
            "corrections": 0,
        }

        plan = self.generate_initial_plan(task)

        for step_num in range(20):
            step = self.get_next_step(plan, step_num)

            if step.get("type") == "final_answer":
                return {"result": step["content"], "success": True}

            # 执行步骤
            result = self.execute_step(step)

            # 检测失败
            detection = self.detector.detect(step, result, context)

            if detection["has_failure"]:
                correction_result = self.handle_failure(
                    step, result, detection, context
                )

                if correction_result["action"] == "abort":
                    return {
                        "result": f"任务执行中止: {correction_result['reason']}",
                        "success": False
                    }

                # 使用修正后的结果
                result = correction_result.get("corrected_result", result)

            # 更新上下文
            context["history"].append({
                "step": step_num,
                "action": step,
                "result": result,
                "failures": detection.get("failures", [])
            })

        return {"result": "达到最大步数", "success": False}

    def handle_failure(self, step: dict, result: dict,
                       detection: dict, context: dict) -> dict:
        """处理检测到的失败"""
        context["corrections"] += 1

        if context["corrections"] > self.max_corrections:
            return {
                "action": "abort",
                "reason": "超过最大修正次数"
            }

        for failure in detection["failures"]:
            print(f"[失败检测] {failure['type']}: {failure['detail']}")

            # 根据失败类型采取不同策略
            match failure["action"]:
                case "retry_with_different_approach":
                    correction = self.retry.retry(
                        json.dumps(step), failure,
                        context["corrections"]
                    )
                    return {"action": "retry", "correction": correction}

                case "self_correct":
                    corrected = self.self_correct(step, result, failure)
                    return {"action": "corrected", "corrected_result": corrected}

                case "continue_with_warning":
                    print(f"[警告] {failure['detail']}")
                    return {"action": "continue"}

        return {"action": "continue"}

    def self_correct(self, step: dict, result: dict, failure: dict) -> dict:
        """让 LLM 自我修正"""
        corrected = self.llm.generate(f"""
你之前的操作出现了问题，请分析并修正。

原始操作: {json.dumps(step, ensure_ascii=False)}
执行结果: {json.dumps(result, ensure_ascii=False)}
失败类型: {failure['type']}
失败详情: {failure['detail']}

请：
1. 分析错误原因
2. 生成修正后的操作
3. 输出修正说明

JSON 输出: {{"analysis": "...", "corrected_action": {{...}}, "explanation": "..."}}
""")
        return json.loads(corrected)
```

## 5. 检查点与回滚

```python
class CheckpointManager:
    """检查点管理，支持回滚"""

    def __init__(self):
        self.checkpoints = []

    def save_checkpoint(self, state: dict, label: str = ""):
        """保存检查点"""
        self.checkpoints.append({
            "id": len(self.checkpoints),
            "label": label,
            "state": deepcopy(state),
            "timestamp": time.time()
        })

    def rollback(self, checkpoint_id: int = None) -> dict:
        """回滚到检查点"""
        if checkpoint_id is None:
            # 回滚到最近的检查点
            checkpoint = self.checkpoints[-1]
        else:
            checkpoint = self.checkpoints[checkpoint_id]

        # 删除该检查点之后的所有状态
        self.checkpoints = [
            c for c in self.checkpoints if c["id"] <= checkpoint_id
        ]

        print(f"[回滚] 回到检查点 #{checkpoint['id']}: {checkpoint['label']}")
        return checkpoint["state"]

class ResilientAgent(SelfCorrectingAgent):
    """带检查点的弹性 Agent"""

    def run(self, task: str) -> dict:
        checkpoint_mgr = CheckpointManager()
        state = self.initialize_state(task)

        for step_num in range(20):
            # 关键步骤前保存检查点
            if self.is_critical_step(state):
                checkpoint_mgr.save_checkpoint(state, f"step_{step_num}")

            try:
                result = self.execute_step(state)
                state = self.update_state(state, result)

            except Exception as e:
                print(f"[异常] 步骤 {step_num}: {e}")

                # 回滚并重试
                state = checkpoint_mgr.rollback()
                state = self.adapt_after_rollback(state, str(e))

        return state
```

## 6. 优雅降级

```python
class GracefulDegradation:
    """优雅降级策略"""

    def __init__(self, fallback_chain: list):
        """
        fallback_chain: 从最理想到最保守的降级链
        [
            {"name": "full_agent", "handler": full_agent_handler},
            {"name": "simplified_agent", "handler": simplified_handler},
            {"name": "template_response", "handler": template_handler},
            {"name": "human_handoff", "handler": human_handler},
        ]
        """
        self.fallback_chain = fallback_chain

    def execute(self, task: str, start_level: int = 0) -> dict:
        """尝试执行，失败则降级"""
        for i, level in enumerate(self.fallback_chain[start_level:], start_level):
            try:
                print(f"[降级] 尝试级别 {i}: {level['name']}")
                result = level["handler"](task)

                if result.get("success"):
                    return {
                        **result,
                        "degradation_level": i,
                        "strategy": level["name"]
                    }

            except Exception as e:
                print(f"[降级] 级别 {i} 失败: {e}")
                continue

        return {
            "success": False,
            "message": "所有降级策略均失败，需要人工介入"
        }
```

## 总结

Agent 的错误恢复核心是**检测-诊断-修正**三步循环。关键实践：**(1) 全面的失败检测** -- 不仅检测显式错误，还要检测循环和漂移；**(2) 分级重试策略** -- 不同错误用不同修正方法；**(3) 检查点与回滚** -- 让 Agent 能够"撤回"错误操作；**(4) 优雅降级** -- 确保即使完全失败也能给出有用响应。
