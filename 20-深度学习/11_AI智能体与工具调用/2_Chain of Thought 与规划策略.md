# 2_Chain of Thought 与规划策略

## 1. Chain of Thought (CoT) 基础

Chain of Thought（思维链）由 Wei et al. (2022) 提出，核心思想是让模型**显式展示中间推理步骤**，而非直接输出答案。

### CoT 的三种触发方式

```python
# 方式 1: Zero-shot CoT — 添加 "Let's think step by step"
prompt = """
问题: 一个商店有 23 个苹果，卖出了 15 个，又进货 8 个，现在有多少个？
让我们一步步思考。
"""

# 方式 2: Few-shot CoT — 提供推理示例
prompt = """
问题: 小明有 5 个苹果，给了小红 2 个，又买了 3 个。
思考: 5 - 2 + 3 = 6
答案: 6 个

问题: 商店有 23 个苹果，卖出 15 个，进货 8 个。
思考:
"""

# 方式 3: Self-CoT — 模型自动生成思维链
prompt = """
请详细展示你的推理过程，然后给出答案。
问题: ...
"""
```

### CoT 的效果分析

```
模型规模 vs CoT 效果：

准确率
  │        ╭── CoT (大模型)
  │       ╱
  │      ╱     ╭── Standard Prompting
  │     ╱     ╱
  │    ╱     ╱
  │   ╱     ╱
  │──╱─────╱──────────→ 模型参数量
  │ ╱     ╱
  │╱     ╱  ← CoT 在小模型上可能退化
  │

关键发现：
- CoT 在 >100B 参数模型上效果显著
- 小模型使用 CoT 可能产生更差结果
- 复杂推理任务 CoT 收益更大
```

## 2. Tree of Thoughts (ToT)

Tree of Thoughts（思维树）由 Yao et al. (2023) 提出，将推理从**线性链**扩展为**树状搜索**。

### 核心思想

```
          问题
         / | \
       T1  T2  T3        ← 生成多个思考方向
      /|\  |  /\
     ...  ...   ...       ← 每个方向继续扩展
     |    |    |
    评估  评估  评估       ← 评估各路径质量
     ↓    ↓    ↓
    剪枝  保留  剪枝       ← 保留最有前景的路径
         |
       最优解
```

### ToT 实现框架

```python
from dataclasses import dataclass, field

@dataclass
class ThoughtNode:
    content: str
    score: float = 0.0
    children: list["ThoughtNode"] = field(default_factory=list)
    parent: "ThoughtNode" | None = None
    depth: int = 0

class TreeOfThoughts:
    def __init__(self, llm, branching_factor: int = 3, max_depth: int = 4):
        self.llm = llm
        self.bf = branching_factor  # 每个节点的分支数
        self.max_depth = max_depth

    def solve(self, problem: str) -> str:
        root = ThoughtNode(content=problem)
        self._expand(root)
        return self._get_best_path(root)

    def _expand(self, node: ThoughtNode):
        if node.depth >= self.max_depth:
            return

        # 生成多个思考方向
        thoughts = self._generate_thoughts(node)

        for thought in thoughts:
            child = ThoughtNode(
                content=thought,
                parent=node,
                depth=node.depth + 1
            )
            # 评估该思考方向的价值
            child.score = self._evaluate(child)
            node.children.append(child)

            # 剪枝：只扩展高分节点
            if child.score > 0.5:
                self._expand(child)

    def _generate_thoughts(self, node: ThoughtNode) -> list[str]:
        prompt = f"""
当前状态: {self._get_path(node)}
请生成 {self.bf} 个不同的下一步思考方向。
每个思考一行，格式："- 思考内容"
"""
        response = self.llm.generate(prompt)
        return self._parse_thoughts(response)

    def _evaluate(self, node: ThoughtNode) -> float:
        """评估当前思考路径的前景 (0-1)"""
        prompt = f"""
思考路径: {self._get_path(node)}
这个推理路径能否解决问题？评分 0-1，并简要说明。
格式: "评分: X.X"
"""
        response = self.llm.generate(prompt)
        score_match = re.search(r"(\d\.?\d*)", response)
        return min(float(score_match.group(1)), 1.0) if score_match else 0.5

    def _get_path(self, node: ThoughtNode) -> str:
        path = []
        current = node
        while current:
            path.append(current.content)
            current = current.parent
        return " → ".join(reversed(path))

    def _get_best_path(self, root: ThoughtNode) -> str:
        """DFS 找到最高分的叶节点"""
        best = {"score": -1, "path": ""}

        def dfs(node):
            if not node.children:
                if node.score > best["score"]:
                    best["score"] = node.score
                    best["path"] = self._get_path(node)
                return
            for child in node.children:
                dfs(child)

        dfs(root)
        return best["path"]
```

### ToT 搜索策略

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| **BFS（广度优先）** | 每层扩展所有节点，评估后保留 Top-K | 搜索空间较小 |
| **DFS（深度优先）** | 沿一条路径深入，失败则回溯 | 有明确终止条件 |
| **Beam Search** | 每层保留固定数量最优路径 | 平衡质量与效率 |
| **MCTS（蒙特卡洛树搜索）** | 模拟评估+UCB 选择 | 大搜索空间、游戏类 |

```python
# Beam Search 变体
class BeamSearchToT(TreeOfThoughts):
    def solve(self, problem: str, beam_width: int = 3) -> str:
        beam = [ThoughtNode(content=problem)]

        for depth in range(self.max_depth):
            candidates = []
            for node in beam:
                thoughts = self._generate_thoughts(node)
                for t in thoughts:
                    child = ThoughtNode(content=t, parent=node, depth=depth+1)
                    child.score = self._evaluate(child)
                    candidates.append(child)

            # 保留 Top-K
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:beam_width]

        return self._get_path(beam[0])
```

## 3. Graph of Thoughts (GoT)

Graph of Thoughts（思维图）由 Besta et al. (2023) 提出，将推理结构从树扩展为**有向图**，支持节点合并和循环。

```
ToT (树结构):                GoT (图结构):
    A                           A
   /|\                         /|\
  B  C  D                      B  C  D
 /\    /\                      | / \ |
E  F  G  H                     E─F   G
                               |     |
                               └──H──┘
                              (节点可合并、循环)
```

### GoT 核心操作

```python
class GraphOfThoughts:
    """支持五种核心操作"""

    def generate(self, node: ThoughtNode, n: int = 3):
        """从一个节点生成多个子节点"""
        pass

    def score(self, node: ThoughtNode) -> float:
        """评估节点质量"""
        pass

    def aggregate(self, nodes: list[ThoughtNode]) -> ThoughtNode:
        """合并多个节点的思路"""
        combined = "\n---\n".join(n.content for n in nodes)
        summary = self.llm.generate(f"综合以下思路:\n{combined}")
        return ThoughtNode(content=summary)

    def refine(self, node: ThoughtNode) -> ThoughtNode:
        """优化已有节点的内容"""
        improved = self.llm.generate(f"改进以下思路:\n{node.content}")
        return ThoughtNode(content=improved)

    def keep_best(self, nodes: list[ThoughtNode], k: int) -> list[ThoughtNode]:
        """保留最优的 k 个节点"""
        return sorted(nodes, key=lambda n: n.score, reverse=True)[:k]
```

## 4. 三种策略对比

```
┌──────────┬────────────┬────────────┬────────────┐
│   维度    │    CoT     │    ToT     │    GoT     │
├──────────┼────────────┼────────────┼────────────┤
│ 结构      │ 线性链     │ 树          │ 有向图      │
│ 分支      │ 无         │ 多分支      │ 任意连接    │
│ 回溯      │ 不支持     │ 支持        │ 支持        │
│ 节点合并  │ 不支持     │ 不支持      │ 支持        │
│ LLM调用  │ 1次        │ O(b^d)     │ O(b^d)     │
│ 延迟      │ 低         │ 中          │ 高          │
│ 适用任务  │ 简单推理    │ 探索性任务   │ 复杂组合    │
└──────────┴────────────┴────────────┴────────────┘
```

## 5. 规划策略在 Agent 中的应用

### 5.1 任务分解 (Task Decomposition)

```python
class TaskPlanner:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, task: str) -> list[dict]:
        """将复杂任务分解为子任务"""
        prompt = f"""
将以下任务分解为可执行的子任务列表。

任务: {task}

输出格式 (JSON):
[
  {{"id": 1, "description": "子任务描述", "depends_on": [], "tool": "需要的工具"}},
  {{"id": 2, "description": "...", "depends_on": [1], "tool": "..."}}
]
"""
        response = self.llm.generate(prompt)
        return json.loads(response)

    def create_execution_plan(self, subtasks: list[dict]) -> list[list[dict]]:
        """拓扑排序，生成可并行执行的批次"""
        from collections import defaultdict, deque

        graph = defaultdict(list)
        in_degree = defaultdict(int)

        for task in subtasks:
            in_degree[task["id"]] = len(task.get("depends_on", []))
            for dep in task.get("depends_on", []):
                graph[dep].append(task["id"])

        # 按层级分组
        batches = []
        remaining = {t["id"]: t for t in subtasks}
        task_map = remaining.copy()

        while remaining:
            # 找到所有入度为 0 的任务
            batch = [task_map[tid] for tid in remaining if in_degree[tid] == 0]
            if not batch:
                break  # 存在循环依赖
            batches.append(batch)

            for task in batch:
                del remaining[task["id"]]
                for neighbor in graph[task["id"]]:
                    in_degree[neighbor] -= 1

        return batches
```

### 5.2 自适应规划

```python
class AdaptivePlanner:
    """根据执行反馈动态调整计划"""

    def plan_and_execute(self, task: str) -> str:
        plan = self.create_initial_plan(task)
        results = {}

        for step in plan:
            result = self.execute_step(step, results)
            results[step["id"]] = result

            # 检查是否需要重新规划
            if self.needs_replanning(step, result):
                remaining = self.get_remaining_steps(plan, step)
                new_plan = self.replan(task, results, remaining)
                plan = self.merge_plans(plan, new_plan, step)

        return self.synthesize_results(results)

    def needs_replanning(self, step: dict, result) -> bool:
        # 失败、结果异常、或发现新信息时重新规划
        return (
            result is None or
            "error" in str(result).lower() or
            step.get("confidence", 1.0) < 0.3
        )
```

## 6. 选择策略指南

```
任务复杂度评估：

简单 (1-2步) ──→ CoT 足够
    │
中等 (3-5步) ──→ ReAct + CoT
    │
复杂 (多步+分支) ──→ ToT
    │
超复杂 (依赖+合并) ──→ GoT

成本敏感场景：
- 优先 CoT / Zero-shot CoT
- 必要时升级到 ToT (限制搜索深度)

准确性优先场景：
- 直接用 ToT/GoT
- 结合 Self-Consistency 多次采样取共识
```

## 7. Self-Consistency：增强 CoT 可靠性

```python
class SelfConsistentCoT:
    """多次采样取多数投票"""

    def __init__(self, llm, n_samples: int = 5):
        self.llm = llm
        self.n_samples = n_samples

    def solve(self, problem: str) -> str:
        answers = []
        for _ in range(self.n_samples):
            response = self.llm.generate(
                f"{problem}\n让我们一步步思考。",
                temperature=0.7  # 提高采样多样性
            )
            answer = self.extract_answer(response)
            answers.append(answer)

        # 多数投票
        from collections import Counter
        most_common = Counter(answers).most_common(1)[0]
        return most_common[0]
```

## 总结

CoT、ToT、GoT 构成了从简单到复杂的**规划策略谱系**。选择依据是：任务复杂度、成本预算、准确性要求。在 Agent 系统中，通常将这些规划策略与 ReAct 循环结合使用 -- 用 CoT/ToT 生成计划，用 ReAct 执行计划。
