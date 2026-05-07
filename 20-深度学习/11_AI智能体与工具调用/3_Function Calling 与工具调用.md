# 3_Function Calling 与工具调用

## 1. Function Calling 概述

Function Calling 是 OpenAI 在 2023 年引入的标准接口，让 LLM 能够**结构化地调用外部函数**。模型不再输出自由文本的工具调用指令，而是输出**符合预定义 schema 的 JSON 参数**。

```
传统方式 (ReAct):
  LLM 输出: "Action: search\nAction Input: 今日天气"
  → 需要解析文本，格式不稳定

Function Calling:
  LLM 输出: {"name": "get_weather", "arguments": {"city": "北京"}}
  → 结构化 JSON，可靠解析
```

## 2. 工具定义 Schema

### OpenAI 格式

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如 '北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位",
                        "default": "celsius"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词"
                    },
                    "num_results": {
                        "type": "integer",
                        "description": "返回结果数量",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    }
]
```

### Anthropic Claude 格式

```python
tools = [
    {
        "name": "get_weather",
        "description": "获取指定城市的当前天气信息",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称"
                }
            },
            "required": ["city"]
        }
    }
]
```

### Schema 设计原则

```
好的工具描述：
✓ 名称清晰动词化: "search_documents" 而非 "doc_tool"
✓ 描述包含使用场景: "当需要查询公司内部文档时使用"
✓ 参数描述包含格式要求: "日期格式 YYYY-MM-DD"
✓ 必需 vs 可选参数明确标注

差的工具描述：
✗ 名称模糊: "tool1", "helper"
✗ 无描述或描述过短: "搜索"
✗ 参数类型不明确
✗ 缺少示例值
```

## 3. Function Calling 完整流程

```python
import openai
import json

class FunctionCallingAgent:
    def __init__(self, model: str = "gpt-4"):
        self.client = openai.OpenAI()
        self.model = model
        self.tools = {}
        self.messages = []

    def register_tool(self, func, schema: dict):
        """注册工具函数"""
        self.tools[schema["function"]["name"]] = func
        return schema

    def chat(self, user_message: str) -> str:
        self.messages.append({"role": "user", "content": user_message})

        while True:
            # 1. 调用 LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=[s for s in self.tool_schemas],
                tool_choice="auto"
            )

            message = response.choices[0].message
            self.messages.append(message)

            # 2. 检查是否需要调用工具
            if not message.tool_calls:
                return message.content

            # 3. 执行工具调用
            for tool_call in message.tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                print(f"[调用] {func_name}({func_args})")

                if func_name in self.tools:
                    result = self.tools[func_name](**func_args)
                else:
                    result = f"错误：未知工具 {func_name}"

                # 4. 将结果返回给模型
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
```

## 4. 并行工具调用

LLM 可以在一次响应中**同时调用多个工具**，前提是这些调用相互独立。

```python
# 用户问题: "北京和上海今天天气怎么样？"
# LLM 一次输出两个工具调用：
response_message = {
    "role": "assistant",
    "tool_calls": [
        {
            "id": "call_1",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "北京"}'
            }
        },
        {
            "id": "call_2",
            "function": {
                "name": "get_weather",
                "arguments": '{"city": "上海"}'
            }
        }
    ]
}

# 两个调用可以并行执行
import asyncio

async def execute_parallel(tool_calls):
    tasks = []
    for call in tool_calls:
        func = tools[call.function.name]
        args = json.loads(call.function.arguments)
        tasks.append(asyncio.to_thread(func, **args))

    results = await asyncio.gather(*tasks)
    return results
```

## 5. 强制工具调用

```python
# 自动选择（默认）
tool_choice = "auto"  # 模型决定是否调用工具

# 强制调用特定工具
tool_choice = {"type": "function", "function": {"name": "get_weather"}}

# 强制调用任意工具
tool_choice = "required"  # 必须调用至少一个工具

# 禁止工具调用
tool_choice = "none"  # 只输出文本

# 应用场景示例
class DataAnalysisAgent:
    def analyze(self, data, question):
        # 第一步：必须调用数据查询工具
        step1 = self.client.chat.completions.create(
            messages=[{"role": "user", "content": f"分析数据: {question}"}],
            tool_choice="required",  # 强制调用工具
            tools=self.data_tools
        )

        # 第二步：允许自由选择（查询结果后再分析）
        step2 = self.client.chat.completions.create(
            messages=self.messages + [step1.choices[0].message, tool_result],
            tool_choice="auto"  # 模型自行决定
        )
```

## 6. 工具调用的类型约束

```python
# 利用 JSON Schema 的类型系统约束参数
parameter_schema = {
    "type": "object",
    "properties": {
        # 字符串约束
        "email": {
            "type": "string",
            "format": "email",
            "pattern": "^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\\.[a-zA-Z0-9-.]+$"
        },
        # 数值约束
        "temperature": {
            "type": "number",
            "minimum": -100,
            "maximum": 100
        },
        # 枚举约束
        "priority": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"]
        },
        # 数组约束
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 5
        },
        # 嵌套对象
        "location": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"},
                "lng": {"type": "number"}
            },
            "required": ["lat", "lng"]
        }
    }
}
```

## 7. 参数验证与错误处理

```python
from pydantic import BaseModel, validator
from typing import Optional

class WeatherQuery(BaseModel):
    city: str
    unit: Optional[str] = "celsius"

    @validator("city")
    def city_must_be_valid(cls, v):
        valid_cities = ["北京", "上海", "广州", "深圳"]
        if v not in valid_cities:
            raise ValueError(f"不支持的城市: {v}")
        return v

class SafeFunctionCaller:
    def call(self, func_name: str, raw_args: str) -> str:
        try:
            # 1. 解析 JSON
            args = json.loads(raw_args)

            # 2. Pydantic 验证
            if func_name == "get_weather":
                validated = WeatherQuery(**args)
            else:
                validated = args

            # 3. 执行调用
            result = self.tools[func_name](**validated.dict())

            # 4. 结果截断（防止 token 爆炸）
            result_str = str(result)
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "\n... (已截断)"

            return result_str

        except json.JSONDecodeError:
            return "错误：参数 JSON 格式无效"
        except ValidationError as e:
            return f"参数验证失败: {e}"
        except Exception as e:
            return f"执行错误: {str(e)}"
```

## 8. 从 Python 函数自动生成 Schema

```python
import inspect
from typing import get_type_hints

def function_to_schema(func) -> dict:
    """将 Python 函数自动转换为 OpenAI 工具 schema"""
    hints = get_type_hints(func)
    sig = inspect.signature(func)

    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        param_type = hints.get(param_name, str)
        prop = {"description": f"参数 {param_name}"}

        # 类型映射
        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object"
        }
        prop["type"] = type_map.get(param_type, "string")

        if param.default == inspect.Parameter.empty:
            required.append(param_name)

        properties[param_name] = prop

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or f"函数 {func.__name__}",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

# 使用示例
def get_stock_price(symbol: str, currency: str = "USD") -> float:
    """获取股票的当前价格"""
    pass

schema = function_to_schema(get_stock_price)
# 自动产生完整的工具定义
```

## 9. 多轮工具调用对话

```python
class ConversationalAgent:
    def __init__(self):
        self.messages = [
            {"role": "system", "content": "你是一个有帮助的助手，可以使用工具完成任务。"}
        ]

    def chat(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})

        max_iterations = 5
        for _ in range(max_iterations):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=self.messages,
                tools=tools
            )

            msg = response.choices[0].message
            self.messages.append(msg)

            if msg.content and not msg.tool_calls:
                return msg.content  # 最终回答

            if msg.tool_calls:
                for call in msg.tool_calls:
                    result = execute_tool(call)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result
                    })

        return "任务完成"
```

## 总结

Function Calling 将 LLM 的工具使用从**文本解析**升级为**结构化调用**，大幅提升了可靠性。核心要点：**(1) Schema 设计要清晰、详细；(2) 参数验证不可省略；(3) 并行调用可提升效率；(4) 结果截断防止 token 爆炸**。这是构建生产级 Agent 的基础设施。
