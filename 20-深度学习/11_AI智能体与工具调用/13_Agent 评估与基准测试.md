# 13_Agent 评估与基准测试

## 1. Agent 评估的挑战

评估 AI Agent 比评估传统 NLP 模型复杂得多，因为 Agent 的行为是**多步的、动态的、依赖环境的**。

```
传统模型评估:              Agent 评估:
Input → Model → Output    Input → Agent ↔ Environment
     ↓                         ↓
  与标签对比                 多维度评估:
  准确率/F1                  - 任务完成度
                            - 过程效率
                            - 安全性
                            - 鲁棒性
```

## 2. 评估维度

```python
class AgentEvaluationFramework:
    """多维度 Agent 评估"""

    def evaluate(self, agent, tasks: list[dict]) -> dict:
        results = {
            "effectiveness": self.eval_effectiveness(agent, tasks),
            "efficiency": self.eval_efficiency(agent, tasks),
            "robustness": self.eval_robustness(agent, tasks),
            "safety": self.eval_safety(agent, tasks),
            "generalization": self.eval_generalization(agent, tasks),
        }

        # 综合评分
        results["overall"] = (
            results["effectiveness"] * 0.35 +
            results["efficiency"] * 0.20 +
            results["robustness"] * 0.20 +
            results["safety"] * 0.15 +
            results["generalization"] * 0.10
        )

        return results

    def eval_effectiveness(self, agent, tasks) -> float:
        """任务完成率"""
        completed = 0
        for task in tasks:
            result = agent.run(task["instruction"])
            if self.check_task_completion(result, task["expected"]):
                completed += 1
        return completed / len(tasks)

    def eval_efficiency(self, agent, tasks) -> float:
        """执行效率"""
        total_steps = 0
        for task in tasks:
            trace = agent.run_with_trace(task["instruction"])
            total_steps += len(trace.steps)

        avg_steps = total_steps / len(tasks)
        # 步骤越少越好，归一化到 0-1
        return max(0, 1 - (avg_steps - 3) / 20)

    def eval_robustness(self, agent, tasks) -> float:
        """鲁棒性：面对干扰时的表现"""
        robust_count = 0
        for task in tasks:
            # 添加噪声输入
            noisy_task = self.add_noise(task)
            result = agent.run(noisy_task["instruction"])
            if self.check_task_completion(result, task["expected"]):
                robust_count += 1
        return robust_count / len(tasks)

    def eval_safety(self, agent, tasks) -> float:
        """安全性"""
        violations = 0
        for task in tasks:
            trace = agent.run_with_trace(task["instruction"])
            for step in trace.steps:
                if self.is_unsafe_action(step):
                    violations += 1
        return max(0, 1 - violations / (len(tasks) * 10))
```

## 3. 主要基准测试

### 3.1 WebArena（网页操作基准）

```
WebArena 评估框架：

环境: 真实网站部署（电商、论坛、地图等）
任务: 812 个自然语言指令
评估: 通过自动化验证检查任务完成度

任务类型示例：
- "在 Reddit 上找到关于 AI 的热门帖子并投票"
- "在电商网站上搜索价格最低的蓝牙耳机"
- "在 GitLab 上创建一个新 issue"
```

```python
class WebArenaEvaluator:
    def evaluate(self, agent, task_config: dict) -> dict:
        """评估 Agent 在 WebArena 任务上的表现"""
        # 设置环境
        env = self.setup_environment(task_config["sites"])

        # 执行任务
        trace = agent.run(
            task_config["instruction"],
            start_url=task_config["start_url"]
        )

        # 验证结果
        is_completed = self.verify(
            task_config["eval_info"],
            env.get_final_state()
        )

        return {
            "task_id": task_config["task_id"],
            "completed": is_completed,
            "steps": len(trace.steps),
            "tokens_used": trace.total_tokens,
            "time_seconds": trace.execution_time,
        }
```

### 3.2 SWE-bench（软件工程基准）

```
SWE-bench 评估：

数据来源: 真实 GitHub Issue + PR
任务: 根据 Issue 描述修复代码
评估: 通过测试套件验证修复

示例任务：
- Issue: "Django QuerySet.values() 在使用 F() 表达式时返回错误结果"
- 期望: Agent 修改 Django 源码，通过相关测试
```

```python
class SWEBenchEvaluator:
    def evaluate(self, coding_agent, instance: dict) -> dict:
        """评估代码修复能力"""
        # 设置仓库环境
        repo = self.clone_repo(instance["repo"], instance["base_commit"])

        # 执行修复
        result = coding_agent.fix_issue(
            issue=instance["issue_text"],
            repo_path=repo.path
        )

        # 运行测试
        test_results = repo.run_tests(instance["test_patch"])

        return {
            "instance_id": instance["instance_id"],
            "resolved": test_results["all_pass"],
            "tests_passed": test_results["passed"],
            "tests_failed": test_results["failed"],
            "patch_diff": result.patch,
        }
```

### 3.3 AgentBench

```
AgentBench 综合评估 8 个环境：

1. 操作系统 (Linux 终端)
2. 数据库 (SQL 操作)
3. 知识图谱 (知识查询)
4. 网页购物 (电商)
5. 网页浏览 (信息检索)
6. 数学推理 (Math)
7. 科学实验 (ALFWorld)
8. 工具使用 (ToolBench)

评估指标：
- 完成率 (Completion Rate)
- 正确率 (Correctness)
- 效率 (Steps/Cost)
```

## 4. 评估方法

### 4.1 过程评估 vs 结果评估

```python
class ProcessEvaluator:
    """评估 Agent 的执行过程"""

    def evaluate_trace(self, trace: AgentTrace, golden_trace: list) -> dict:
        # 工具调用准确率
        tool_accuracy = self.compare_tool_calls(
            trace.tool_calls, golden_trace
        )

        # 思维链质量
        reasoning_score = self.evaluate_reasoning(trace.thoughts)

        # 操作序列相似度
        sequence_similarity = self.sequence_match(
            trace.actions, [s["action"] for s in golden_trace]
        )

        return {
            "tool_accuracy": tool_accuracy,
            "reasoning_quality": reasoning_score,
            "sequence_similarity": sequence_similarity,
        }

class OutcomeEvaluator:
    """仅评估最终结果"""

    def evaluate(self, agent_output: str, expected: str) -> dict:
        # 精确匹配
        exact_match = agent_output.strip() == expected.strip()

        # 语义相似度
        similarity = cosine_similarity(
            self.embed(agent_output),
            self.embed(expected)
        )

        # LLM 评分
        llm_score = self.llm_judge(agent_output, expected)

        return {
            "exact_match": exact_match,
            "semantic_similarity": similarity,
            "llm_score": llm_score,
        }
```

### 4.2 LLM-as-Judge

```python
class LLMJudge:
    """使用 LLM 作为评估者"""

    def judge(self, task: str, agent_output: str,
              expected: str = None) -> dict:
        prompt = f"""
作为 AI Agent 评估专家，评估以下 Agent 的表现：

任务: {task}
Agent 输出: {agent_output}
{"期望结果: " + expected if expected else ""}

评分维度 (1-10):
1. 任务完成度: 是否完成了指定任务？
2. 结果准确性: 输出是否正确？
3. 表达清晰度: 输出是否易于理解？
4. 效率: 是否用最少步骤完成？

输出 JSON:
{{"completion": X, "accuracy": X, "clarity": X, "efficiency": X, "overall": X, "comments": "..."}}
"""
        return json.loads(self.llm.generate(prompt))
```

## 5. 自动化评估管道

```python
class AutoEvaluationPipeline:
    """自动化评估管道"""

    def __init__(self, agent, benchmarks: dict):
        self.agent = agent
        self.benchmarks = benchmarks

    def run_full_evaluation(self) -> dict:
        all_results = {}

        for benchmark_name, benchmark in self.benchmarks.items():
            print(f"\n=== 评估 {benchmark_name} ===")
            results = []

            for task in benchmark.tasks:
                # 执行任务
                start_time = time.time()
                trace = self.agent.run_with_trace(task.instruction)
                elapsed = time.time() - start_time

                # 多维度评估
                eval_result = {
                    "task_id": task.id,
                    "completed": self.verify_completion(trace, task),
                    "steps": len(trace.steps),
                    "time": elapsed,
                    "tokens": trace.total_tokens,
                    "cost": trace.estimated_cost,
                    "safety_violations": self.check_safety(trace),
                }

                results.append(eval_result)
                print(f"  任务 {task.id}: {'通过' if eval_result['completed'] else '失败'} "
                      f"({eval_result['steps']}步, {eval_result['time']:.1f}s)")

            # 汇总
            all_results[benchmark_name] = {
                "total_tasks": len(results),
                "completed": sum(1 for r in results if r["completed"]),
                "completion_rate": sum(1 for r in results if r["completed"]) / len(results),
                "avg_steps": sum(r["steps"] for r in results) / len(results),
                "avg_time": sum(r["time"] for r in results) / len(results),
                "total_cost": sum(r["cost"] for r in results),
                "details": results,
            }

        return all_results
```

## 6. 对比总结

| 基准 | 环境 | 任务数 | 评估重点 |
|------|------|--------|----------|
| **WebArena** | 真实网站 | 812 | 网页交互能力 |
| **SWE-bench** | GitHub仓库 | 2294 | 代码修复能力 |
| **AgentBench** | 8种环境 | 749 | 综合能力 |
| **GAIA** | 通用 | 466 | 通用推理+工具使用 |
| **ToolBench** | API调用 | 16000+ | 工具调用能力 |
| **OSWorld** | 操作系统 | 369 | OS操作能力 |

## 总结

Agent 评估需要**多维度、过程+结果结合**的方法。WebArena 和 SWE-bench 是目前最受关注的两个基准。核心挑战在于：(1) 评估环境的真实性；(2) 评估指标的全面性；(3) 自动化评估的可靠性。LLM-as-Judge 是一个有前景但需要注意偏差的评估方法。
