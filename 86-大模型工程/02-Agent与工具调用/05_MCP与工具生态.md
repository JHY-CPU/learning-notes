# MCP与工具生态 - Agent与工具调用

*Model Context Protocol 架构、工具规范设计、MCP vs Function Calling 对比及集成实践*

MCP 解决的核心问题

| 问题 | MCP 之前 | MCP 方案 |
| --- | --- | --- |
| 工具集成碎片化 | 每个应用自定义接口 | 统一协议标准 |
| 数据源连接复杂 | 为每个数据源写代码 | 通用 MCP Server |
| 上下文管理分散 | 应用内硬编码 | 标准化上下文提供 |
| 生态难以互通 | 工具无法跨应用复用 | 一次实现，处处可用 |

MCP Server 提供的三种能力

| 能力 | 说明 | 示例 |
| --- | --- | --- |
| **Tools（工具）** | 可被 LLM 调用的函数/操作 | 读写文件、执行 SQL、调用 API |
| **Resources（资源）** | 可被读取的数据/上下文 | 文件内容、数据库 schema、API 文档 |
| **Prompts（提示模板）** | 预定义的高质量提示词 | 代码审查模板、分析报告模板 |

MCP 传输方式对比

| 传输方式 | 原理 | 适用场景 |
| --- | --- | --- |
| **stdio** | 通过标准输入输出通信 | 本地进程（最常用） |
| **SSE (Server-Sent Events)** | HTTP 长连接推送 | 远程 Server、Web 集成 |
| **Streamable HTTP** | HTTP 请求+流式响应 | 生产部署、负载均衡 |

MCP 与 Function Calling 深度对比

| 维度 | Function Calling | MCP |
| --- | --- | --- |
| **标准性** | 各厂商自定义格式 | 开放统一标准 |
| **工具发现** | 应用启动时静态注册 | 运行时动态发现（list_tools） |
| **数据源接入** | 需要单独实现 | Resources 原生支持 |
| **跨应用复用** | 工具定义绑定应用 | 一次实现处处可用 |
| **传输方式** | API 调用（HTTP） | stdio / SSE / HTTP |
| **状态管理** | 无状态 | 支持会话级状态 |
| **安全性** | 应用层控制 | 协议级权限控制 |
| **适用场景** | 简单的工具调用 | 复杂工具生态集成 |

主流 MCP Server 生态

| 类别 | MCP Server | 功能 |
| --- | --- | --- |
| 文件系统 | @anthropic-ai/mcp-server-filesystem | 安全的文件读写操作 |
| 版本控制 | @anthropic-ai/mcp-server-github | GitHub 操作（Issue/PR/Code） |
| 数据库 | @anthropic-ai/mcp-server-postgres | PostgreSQL 查询 |
| 数据库 | @anthropic-ai/mcp-server-sqlite | SQLite 操作 |
| 浏览器 | 浏览器自动化 |
| 搜索 | @anthropic-ai/mcp-server-brave-search | Web 搜索 |
| 记忆 | @anthropic-ai/mcp-server-memory | 知识图谱记忆 |
| Slack | @anthropic-ai/mcp-server-slack | Slack 消息操作 |
| Google Maps | @anthropic-ai/mcp-server-google-maps | 地图和位置服务 |
| Sentry | @anthropic-ai/mcp-server-sentry | 错误监控和追踪 |


<!-- Converted from: 05_MCP与工具生态.html -->
