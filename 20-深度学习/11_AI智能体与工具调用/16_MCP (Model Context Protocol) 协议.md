# 16_MCP (Model Context Protocol) 协议

## 1. MCP 概述

Model Context Protocol (MCP) 是 Anthropic 于 2024 年提出的**开放标准协议**，旨在为 AI Agent 提供**标准化的工具和数据源连接方式**。类比 USB-C 接口统一了设备连接，MCP 试图统一 Agent 与外部世界的交互。

```
没有 MCP (Before):
  Agent ←──自定义接口──→ 工具A
  Agent ←──自定义接口──→ 工具B
  Agent ←──自定义接口──→ 数据库
  (每对接一个工具都要写专门的适配器)

有了 MCP (After):
  Agent ←──MCP协议──→ MCP Server A (工具集合)
  Agent ←──MCP协议──→ MCP Server B (数据源)
  Agent ←──MCP协议──→ MCP Server C (API服务)
  (标准化接口，即插即用)
```

## 2. MCP 架构

```
┌─────────────────────────────────────────────┐
│                MCP Client                    │
│           (AI 应用/Agent)                    │
│  ┌──────────────────────────────────────┐  │
│  │  MCP SDK (协议实现)                    │  │
│  │  - Tool 调用                          │  │
│  │  - Resource 读取                      │  │
│  │  - Prompt 模板获取                    │  │
│  │  - 采样请求转发                       │  │
│  └──────────────┬───────────────────────┘  │
└─────────────────┼───────────────────────────┘
                  │ JSON-RPC 2.0
┌─────────────────┼───────────────────────────┐
│  MCP Server  ←──┘                            │
│  ┌──────────────────────────────────────┐  │
│  │  Tools: 暴露的可调用函数              │  │
│  │  Resources: 可读取的数据源            │  │
│  │  Prompts: 预定义的提示词模板          │  │
│  └──────────────────────────────────────┘  │
│  (文件系统/Git/数据库/API/自定义服务)       │
└─────────────────────────────────────────────┘
```

## 3. MCP Server 实现

### 3.1 基础 Server

```python
from mcp.server import Server
from mcp.types import Tool, TextContent, Resource
import mcp.server.stdio

# 创建 MCP Server
server = Server("my-tools-server")

# 注册工具
@server.tool("read_file")
async def read_file(path: str) -> str:
    """读取指定路径的文件内容"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

@server.tool("write_file")
async def write_file(path: str, content: str) -> str:
    """将内容写入指定路径的文件"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"已写入 {path} ({len(content)} 字符)"

@server.tool("list_directory")
async def list_directory(path: str = ".") -> str:
    """列出目录内容"""
    import os
    items = []
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        type_str = "DIR" if os.path.isdir(item_path) else "FILE"
        items.append(f"[{type_str}] {item}")
    return "\n".join(items)

# 注册资源
@server.resource("file:///{path}")
async def get_file_resource(path: str) -> str:
    """提供文件内容作为资源"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

# 注册提示词模板
@server.prompt("code_review")
async def code_review_prompt(file_path: str) -> str:
    """代码审查提示词模板"""
    code = await read_file(file_path)
    return f"""请审查以下代码：

```python
{code}
```

审查维度：
1. 代码质量和可读性
2. 潜在 bug
3. 性能问题
4. 安全漏洞
5. 改进建议
"""

# 启动 Server
async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### 3.2 复杂工具定义

```python
from mcp.types import ToolAnnotations

@server.tool(
    "database_query",
    annotations=ToolAnnotations(
        title="数据库查询",
        readOnlyHint=True,        # 只读操作
        destructiveHint=False,    # 非破坏性
        openWorldHint=True,       # 访问外部世界
    )
)
async def database_query(
    query: str,
    database: str = "default",
    limit: int = 100
) -> str:
    """执行 SQL 查询（只读）

    参数:
        query: SQL SELECT 查询语句
        database: 数据库名称
        limit: 最大返回行数
    """
    # 安全检查
    if not query.strip().upper().startswith("SELECT"):
        return "错误：只允许 SELECT 查询"

    # 执行查询
    results = await db.execute(query, database, limit)
    return format_results(results)
```

## 4. MCP Client 实现

```python
from mcp import ClientSession
from mcp.client.stdio import stdio_client

class MCPAgent:
    """使用 MCP 的 Agent"""

    def __init__(self, llm):
        self.llm = llm
        self.servers = {}  # name -> server_config
        self.available_tools = []

    async def connect_server(self, name: str, command: list[str]):
        """连接 MCP Server"""
        server_params = StdioServerParameters(command=command)

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                # 发现工具
                tools = await session.list_tools()
                for tool in tools.tools:
                    self.available_tools.append({
                        "name": f"{name}.{tool.name}",
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                        "session": session,
                        "original_name": tool.name,
                    })

                # 发现资源
                resources = await session.list_resources()
                self.servers[name] = {
                    "session": session,
                    "tools": tools,
                    "resources": resources
                }

    async def run(self, user_input: str) -> str:
        """Agent 主循环"""
        messages = [{"role": "user", "content": user_input}]

        while True:
            # LLM 决策
            response = await self.llm.chat(
                messages,
                tools=self._format_tools_for_llm()
            )

            if response.content:
                return response.content

            # 执行工具调用
            for call in response.tool_calls:
                result = await self._execute_tool(call)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result
                })

    async def _execute_tool(self, tool_call) -> str:
        """执行 MCP 工具调用"""
        full_name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)

        # 查找对应的 server 和工具
        for tool in self.available_tools:
            if tool["name"] == full_name:
                result = await tool["session"].call_tool(
                    tool["original_name"], args
                )
                return result.content[0].text

        return f"工具 {full_name} 不存在"
```

## 5. MCP Server 生态

```
官方和社区 MCP Server：

文件系统:
  - filesystem: 文件读写、搜索
  - git: 版本控制操作

数据库:
  - sqlite: SQLite 操作
  - postgres: PostgreSQL 操作
  - mysql: MySQL 操作

开发工具:
  - github: GitHub API (Issues, PR, Code)
  - gitlab: GitLab API
  - npm: 包管理查询

搜索:
  - brave-search: 网页搜索
  - puppeteer: 浏览器自动化

通讯:
  - slack: Slack 消息
  - email: 邮件收发

其他:
  - fetch: HTTP 请求
  - memory: 持久化记忆
  - sequential-thinking: 推理链
```

### 配置示例 (Claude Desktop)

```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-filesystem", "/home/user/docs"]
    },
    "github": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-github"],
      "env": {
        "GITHUB_TOKEN": "${GITHUB_TOKEN}"
      }
    },
    "sqlite": {
      "command": "npx",
      "args": ["-y", "@anthropic/mcp-server-sqlite", "/path/to/db.sqlite"]
    }
  }
}
```

## 6. MCP vs Function Calling 对比

```
┌──────────────┬──────────────────┬────────────────────────┐
│     维度     │  Function Calling │     MCP                │
├──────────────┼──────────────────┼────────────────────────┤
│ 标准化       │ 厂商私有格式     │ 开放标准协议            │
│ 工具发现     │ 静态配置         │ 动态发现               │
│ 工具更新     │ 需重新配置       │ Server 端热更新        │
│ 资源访问     │ 不支持           │ 支持 Resources         │
│ 提示词模板   │ 不支持           │ 支持 Prompts           │
│ 双向通信     │ 单向调用         │ 支持采样请求           │
│ 跨平台       │ 绑定特定 Provider│ Client 可对接任意 Server│
│ 生态         │ 各厂独立         │ 社区共享               │
└──────────────┴──────────────────┴────────────────────────┘
```

## 7. 安全考量

```python
class MCPSecurityLayer:
    """MCP 安全层"""

    def __init__(self, permission_config: dict):
        self.config = permission_config

    def validate_tool_call(self, server: str, tool: str,
                           args: dict) -> tuple[bool, str]:
        """验证工具调用权限"""
        # 检查 Server 白名单
        if server not in self.config.get("allowed_servers", []):
            return False, f"服务器 {server} 未授权"

        # 检查工具权限
        tool_key = f"{server}.{tool}"
        tool_config = self.config.get("tools", {}).get(tool_key, {})

        if not tool_config.get("allowed", False):
            return False, f"工具 {tool_key} 未授权"

        # 检查参数安全
        if tool_config.get("read_only") and self._is_write_operation(args):
            return False, f"工具 {tool_key} 配置为只读"

        return True, "OK"
```

## 总结

MCP 是 Agent 工具生态的**标准化基础设施**，核心价值在于：**(1) 统一接口** -- 一个协议对接所有工具；**(2) 动态发现** -- 工具增减无需改代码；**(3) 社区生态** -- 共享 MCP Server 减少重复开发。MCP 正在成为 Agent 工具调用的事实标准。
