# 14_SGLang 结构化生成优化

## 1. SGLang 概述

SGLang (Structured Generation Language) 是 UC Berkeley 开发的**高效 LLM 推理框架**，核心创新是 RadixAttention 和结构化解码。

```
SGLang 核心特性:

1. RadixAttention
   前缀感知的 KV Cache 复用 (类似 Radix Tree)

2. 结构化解码
   约束解码 (JSON schema, regex, CFG)

3. 零开销批处理调度
   高效管理并发请求

4. 前端语言
   Python 风格的提示词编排 DSL
```

## 2. RadixAttention

### 2.1 前缀缓存问题

```
传统前缀缓存:
  用 hash(前缀) 作为 key
  问题: 无法处理部分匹配

场景:
  Request 1: [System Prompt (1000 tokens)] + "什么是量子计算"
  Request 2: [System Prompt (1000 tokens)] + "什么是深度学习"
  Request 3: [System Prompt (1000 tokens)] + "什么是量子计算的最新进展"

  Request 1 和 2 共享 1000 tokens 的前缀
  Request 1 和 3 共享 1000 + 5 tokens 的前缀
  → 需要层级化的缓存管理
```

### 2.2 Radix Tree 结构

```
RadixAttention 的 Radix Tree:

                    root
                   /    \
          [System Prompt]  (1000 tokens, 缓存)
            /         \
    [什么是量子计算]    [什么是深度学习]
        |                  |
    [的最新进展]        (独立缓存)
        |
    (独立缓存)

优势:
  ✓ 支持部分前缀匹配
  ✓ 前缀越长，缓存命中越多
  ✓ 自动管理缓存生命周期
```

```python
class RadixTreeNode:
    """Radix Tree 节点"""

    def __init__(self, token_ids: list, kv_cache=None):
        self.tokens = token_ids      # 该节点对应的 token 序列
        self.kv_cache = kv_cache     # 对应的 KV Cache
        self.children = []           # 子节点
        self.ref_count = 0           # 引用计数
        self.last_access = time.time()

class RadixAttentionCache:
    """RadixAttention 前缀缓存管理"""

    def __init__(self):
        self.root = RadixTreeNode([])

    def match_prefix(self, token_ids: list) -> tuple:
        """
        匹配最长前缀
        返回: (匹配的 KV Cache, 匹配长度, 剩余 token)
        """
        matched_len = 0
        current = self.root
        remaining = list(token_ids)

        while remaining and current.children:
            found = False
            for child in current.children:
                # 尝试匹配子节点
                common = self._common_prefix(remaining, child.tokens)
                if common > 0:
                    matched_len += common
                    remaining = remaining[common:]
                    if common == len(child.tokens):
                        # 完全匹配子节点
                        current = child
                    found = True
                    break
            if not found:
                break

        current.last_access = time.time()
        current.ref_count += 1
        return current.kv_cache, matched_len, remaining

    def insert(self, token_ids: list, kv_cache):
        """插入新的前缀缓存"""
        # 实现 radix tree 插入逻辑
        pass
```

## 3. 结构化解码

```python
"""
SGLang 结构化解码: 在生成时强制约束输出格式

支持的约束类型:
1. JSON Schema: 输出必须符合 JSON schema
2. 正则表达式: 输出必须匹配 regex
3. 上下文无关文法: 输出必须符合 CFG
"""

import sglang as sgl

# 使用 JSON Schema 约束
@sgl.function
def extract_person_info(s, text):
    s += sgl.system("你是一个信息提取助手。")
    s += sgl.user(f"从以下文本中提取人物信息:\n{text}")
    s += sgl.assistant(sgl.gen("result", regex=r'\{.*\}'))

# 使用 JSON Schema
@sgl.function
def structured_extraction(s, text):
    s += sgl.user(f"从文本提取信息: {text}")
    s += sgl.assistant(
        sgl.gen("result", json_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"}
            },
            "required": ["name", "age", "city"]
        })
    )

# 运行
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")
sgl.set_default_backend(runtime)

result = structured_extraction.run(text="张三，25岁，住在北京")
print(result["result"])
# 必然输出: {"name": "张三", "age": 25, "city": "北京"}
```

## 4. SGLang 前端语言

```python
import sglang as sgl

# 1. 基础提示词编排
@sgl.function
def simple_qa(s, question):
    s += sgl.system("你是一个有帮助的助手。")
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=256))

# 2. 多轮对话
@sgl.function
def multi_turn(s, questions):
    s += sgl.system("你是一个专家。")
    for q in questions:
        s += sgl.user(q)
        s += sgl.assistant(sgl.gen("answer_" + q[:10], max_tokens=128))

# 3. 并行生成
@sgl.function
def parallel_generation(s, question):
    s += sgl.user(f"关于'{question}'，请从三个角度回答:")
    s += sgl.assistant("技术角度: " + sgl.gen("tech", max_tokens=100))
    s += sgl.assistant("经济角度: " + sgl.gen("econ", max_tokens=100))
    s += sgl.assistant("社会角度: " + sgl.gen("social", max_tokens=100))

# 4. 分支与循环
@sgl.function
def branching(s, task):
    s += sgl.user(f"判断任务类型: {task}")
    s += sgl.assistant(sgl.gen("task_type", choices=["简单", "复杂"]))

    # 根据结果分支
    if s["task_type"] == "简单":
        s += sgl.user("直接给出答案")
    else:
        s += sgl.user("请详细分析并给出步骤")
    s += sgl.assistant(sgl.gen("result", max_tokens=512))
```

## 5. 性能优化

```python
# SGLang 性能特性

"""
RadixAttention 带来的加速:

场景: 100 个请求共享 500 token 的 system prompt

无前缀缓存:
  每个请求都需要处理 500 tokens 的 prompt
  总 Prefill: 100 × 500 = 50,000 token 处理

RadixAttention:
  第 1 个请求: 处理 500 tokens, 缓存
  后续请求: 直接复用缓存, 0 tokens 处理
  总 Prefill: 500 tokens 处理

加速比: 100x (对于前缀共享场景)
"""

# SGLang 服务部署
# python -m sglang.launch_server \
#     --model-path meta-llama/Llama-2-7b-hf \
#     --port 30000 \
#     --tp 1

# OpenAI 兼容 API
import openai
client = openai.OpenAI(base_url="http://localhost:30000/v1", api_key="none")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Hello"}],
)
```

## 6. 约束解码的实现原理

```
约束解码 (Constrained Decoding):

步骤:
1. 定义合法 token 集合 (基于 schema/regex)
2. 在每步解码时，mask 掉不合法的 token
3. 只从合法 token 中采样

JSON Schema 约束示例:
  Schema: {"name": string, "age": int}
  生成过程:
    Step 1: 只能生成 "{"           ✓
    Step 2: 只能生成 '"'           ✓
    Step 3: 只能生成 "name"        ✓
    Step 4: 只能生成 '"' → ':'     ✓
    Step 5: 生成字符串值            ✓
    ...

  实现: 使用有限状态自动机 (FSM) 追踪当前合法 token
```

## 7. SGLang vs vLLM 对比

```
┌──────────────┬────────────────┬────────────────┐
│    维度       │    SGLang      │     vLLM       │
├──────────────┼────────────────┼────────────────┤
│ 前缀缓存      │ RadixAttention │ PagedAttention │
│ 缓存粒度      │ 前缀树(灵活)   │ Block(固定)    │
│ 结构化解码    │ 内置支持       │ 需外部库       │
│ 前端语言      │ Python DSL     │ 无             │
│ 吞吐量        │ 优秀           │ 优秀           │
│ 生态成熟度    │ 较新           │ 更成熟         │
│ 适用场景      │ 复杂编排       │ 通用推理       │
└──────────────┴────────────────┴────────────────┘
```

## 总结

SGLang 通过 **RadixAttention 实现高效的前缀缓存复用**，通过**结构化解码确保输出格式正确**，通过**Python DSL 简化复杂提示词编排**。对于需要大量前缀共享和结构化输出的场景，SGLang 是比 vLLM 更好的选择。
