# 4_Tool Use 范式与工具设计

## 1. 工具的本质

工具是 Agent 与外部世界交互的**接口抽象**。良好的工具设计直接决定 Agent 的能力上限和可靠性。

```
Agent 能力 = LLM 推理能力 × 工具生态丰富度 × 工具使用准确性

类比：
- LLM = 大脑（推理、规划）
- 工具 = 手和脚（执行操作）
- 工具描述 = 使用说明书（让大脑知道怎么用）
```

## 2. 工具描述工程 (Tool Description Engineering)

### 2.1 描述质量直接影响选择准确率

```python
# ❌ 差的工具描述
bad_tool = {
    "name": "query",
    "description": "查询数据",
    "parameters": {
        "q": {"type": "string", "description": "查询内容"}
    }
}

# ✅ 好的工具描述
good_tool = {
    "name": "search_knowledge_base",
    "description": (
        "搜索公司内部知识库，查找技术文档、政策文件和FAQ。"
        "适用于：产品功能说明、技术规范、公司政策查询。"
        "不适用于：实时新闻、外部网页搜索。"
        "返回相关文档的标题、摘要和链接。"
    ),
    "parameters": {
        "query": {
            "type": "string",
            "description": "搜索关键词或自然语言问题"
        },
        "category": {
            "type": "string",
            "enum": ["tech", "policy", "faq", "all"],
            "description": "限定搜索范围的文档类别",
            "default": "all"
        },
        "max_results": {
            "type": "integer",
            "description": "返回的最大结果数量",
            "default": 5
        }
    },
    "required": ["query"]
}
```

### 2.2 描述编写 Checklist

```
工具描述检查清单：
□ 名称使用动词+名词格式 (search_files, create_report)
□ 描述说明"什么时候用"和"什么时候不用"
□ 描述说明输入要求和输出格式
□ 每个参数都有清晰的 description
□ 枚举类型列出所有可选值
□ 必需/可选参数明确区分
□ 提供合理的 default 值
□ 复杂参数提供示例值
□ 错误场景说明（如：城市不存在时返回什么）
```

## 3. 工具分类体系

```
┌─────────────────────────────────────────────────┐
│                 工具分类                          │
├──────────────┬──────────────────────────────────┤
│ 信息获取类    │ search, read_file, api_get       │
│ 数据操作类    │ create, update, delete           │
│ 计算分析类    │ calculate, analyze, transform    │
│ 通信交互类    │ send_email, notify, ask_human    │
│ 代码执行类    │ execute_code, run_command        │
│ 状态管理类    │ save_state, load_state, memory   │
└──────────────┴──────────────────────────────────┘
```

## 4. 工具选择策略

### 4.1 基于 LLM 的选择

```python
class ToolSelector:
    """让 LLM 根据任务选择合适的工具"""

    def __init__(self, llm, tools: list[dict]):
        self.llm = llm
        self.tools = tools

    def select(self, task: str, context: str = "") -> list[dict]:
        # 步骤1: 粗筛 — 基于描述匹配
        candidates = self.coarse_filter(task)

        # 步骤2: 精选 — LLM 判断最相关的工具
        if len(candidates) > 5:
            selected = self.llm_select(task, candidates)
            return selected

        return candidates

    def coarse_filter(self, task: str) -> list[dict]:
        """基于关键词和嵌入的快速过滤"""
        task_embedding = self.embed(task)
        scored = []
        for tool in self.tools:
            tool_embedding = self.embed(tool["description"])
            similarity = cosine_similarity(task_embedding, tool_embedding)
            if similarity > 0.3:
                scored.append((similarity, tool))

        scored.sort(reverse=True)
        return [t for _, t in scored[:8]]

    def llm_select(self, task: str, candidates: list[dict]) -> list[dict]:
        """LLM 精选最相关的工具"""
        tool_list = "\n".join(
            f"- {t['name']}: {t['description']}"
            for t in candidates
        )
        prompt = f"""
任务: {task}

可用工具:
{tool_list}

选择完成此任务所需的工具（最多3个），输出工具名称列表。
"""
        response = self.llm.generate(prompt)
        selected_names = self.parse_tool_names(response)
        return [t for t in candidates if t["name"] in selected_names]
```

### 4.2 工具路由 (Tool Routing)

```python
class ToolRouter:
    """基于规则的工具路由"""

    def __init__(self):
        self.routes = []

    def add_route(self, pattern: str, tool_name: str, priority: int = 0):
        self.routes.append((pattern, tool_name, priority))
        self.routes.sort(key=lambda x: x[2], reverse=True)

    def route(self, task: str) -> str | None:
        for pattern, tool_name, _ in self.routes:
            if re.search(pattern, task, re.IGNORECASE):
                return tool_name
        return None  # 交给 LLM 选择

# 使用示例
router = ToolRouter()
router.add_route(r"计算|等于|求解", "calculator", priority=10)
router.add_route(r"搜索|查找|查询", "web_search", priority=5)
router.add_route(r"邮件|发送给", "send_email", priority=5)
```

## 5. 工具错误处理

### 5.1 错误分类

```python
class ToolError(Exception):
    """工具错误基类"""
    pass

class ToolNotFoundError(ToolError):
    """工具不存在"""
    pass

class InvalidArgumentsError(ToolError):
    """参数无效"""
    pass

class ExecutionError(ToolError):
    """执行过程中出错"""
    pass

class TimeoutError(ToolError):
    """执行超时"""
    pass

class RateLimitError(ToolError):
    """频率限制"""
    pass
```

### 5.2 带重试的工具执行

```python
import time
from functools import wraps

def with_retry(max_retries: int = 3, backoff: float = 1.0):
    """工具执行装饰器：自动重试 + 指数退避"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except RateLimitError as e:
                    wait = backoff * (2 ** attempt)
                    print(f"频率限制，等待 {wait}s 后重试...")
                    time.sleep(wait)
                    last_error = e
                except ExecutionError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(backoff)
                except InvalidArgumentsError:
                    raise  # 参数错误不重试

            raise last_error
        return wrapper
    return decorator

# 使用
class SafeToolExecutor:
    @with_retry(max_retries=3)
    def execute(self, tool_name: str, args: dict) -> str:
        tool = self.tools.get(tool_name)
        if not tool:
            raise ToolNotFoundError(f"工具 {tool_name} 不存在")
        return tool(**args)
```

### 5.3 错误信息回传

```python
def execute_with_feedback(self, tool_name: str, args: dict) -> str:
    """执行工具并将错误信息反馈给 LLM"""
    try:
        result = self.tools[tool_name](**args)
        return f"工具 {tool_name} 执行成功:\n{result}"

    except InvalidArgumentsError as e:
        # 告诉 LLM 参数哪里错了
        return (
            f"工具 {tool_name} 参数错误: {str(e)}\n"
            f"请检查参数格式，参考 schema:\n"
            f"{self.get_tool_schema(tool_name)}"
        )

    except ExecutionError as e:
        # 提供替代建议
        alternatives = self.suggest_alternatives(tool_name, args)
        return (
            f"工具 {tool_name} 执行失败: {str(e)}\n"
            f"建议尝试: {alternatives}"
        )
```

## 6. 工具组合与管道

```python
class ToolPipeline:
    """将多个工具串联成管道"""

    def __init__(self):
        self.steps = []

    def add_step(self, tool_name: str, arg_mapping: dict):
        """
        arg_mapping: 将上一步输出映射到下一步输入
        例如: {"query": "$.output.search_term"}
        """
        self.steps.append({"tool": tool_name, "args": arg_mapping})

    def execute(self, initial_input: dict) -> dict:
        context = {"input": initial_input}

        for i, step in enumerate(self.steps):
            # 解析参数映射
            resolved_args = {}
            for arg_name, arg_path in step["args"].items():
                resolved_args[arg_name] = self.resolve(arg_path, context)

            # 执行工具
            result = self.executor.execute(step["tool"], resolved_args)
            context[f"step_{i}"] = {"output": result}

        return context

# 使用示例：搜索 -> 摘要 -> 翻译
pipeline = ToolPipeline()
pipeline.add_step("web_search", {"query": "$.input.query"})
pipeline.add_step("summarize", {"text": "$.step_0.output"})
pipeline.add_step("translate", {
    "text": "$.step_1.output",
    "target_lang": "$.input.lang"
})
```

## 7. 工具的安全边界

```python
class SandboxedTool:
    """带安全边界的工具包装器"""

    def __init__(self, func, config: dict):
        self.func = func
        self.config = config

    def __call__(self, **kwargs):
        # 1. 参数白名单检查
        for key in kwargs:
            if key not in self.config.get("allowed_params", []):
                raise SecurityError(f"不允许的参数: {key}")

        # 2. 参数值检查
        if "max_value" in self.config:
            for param, max_val in self.config["max_value"].items():
                if param in kwargs and kwargs[param] > max_val:
                    raise SecurityError(f"{param} 超过最大值 {max_val}")

        # 3. 执行超时控制
        import signal

        def timeout_handler(signum, frame):
            raise TimeoutError("工具执行超时")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.config.get("timeout", 30))

        try:
            result = self.func(**kwargs)
        finally:
            signal.alarm(0)

        # 4. 输出大小限制
        result_str = str(result)
        max_len = self.config.get("max_output_length", 10000)
        if len(result_str) > max_len:
            result_str = result_str[:max_len] + "\n... (已截断)"

        return result_str
```

## 8. 工具评估指标

```python
class ToolEvaluator:
    """评估工具质量和使用效果"""

    def evaluate(self, tool, test_cases: list[dict]) -> dict:
        results = {
            "accuracy": 0,      # 参数准确率
            "success_rate": 0,   # 成功率
            "avg_latency": 0,    # 平均延迟
            "error_types": {},   # 错误分类统计
        }

        latencies = []
        successes = 0

        for case in test_cases:
            start = time.time()
            try:
                output = tool(**case["input"])
                latency = time.time() - start
                latencies.append(latency)

                if self.check_output(output, case["expected"]):
                    successes += 1

            except Exception as e:
                error_type = type(e).__name__
                results["error_types"][error_type] = \
                    results["error_types"].get(error_type, 0) + 1

        results["success_rate"] = successes / len(test_cases)
        results["avg_latency"] = sum(latencies) / len(latencies)
        return results
```

## 总结

工具设计的核心原则：**(1) 描述要详尽** -- 模型只能通过描述理解工具；**(2) 错误要友好** -- 错误信息帮助模型自我纠正；**(3) 安全要有界** -- 所有外部调用都要有防护。好的工具设计是 Agent 可靠性的基石。
