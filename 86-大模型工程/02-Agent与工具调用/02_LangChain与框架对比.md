# LangChain与框架对比 - Agent与工具调用

*深入理解 LangChain Chains/Agents/Tools 核心概念，对比 LlamaIndex、Semantic Kernel 等主流 LLM 应用框架，明确各自适用场景*

四大 LLM 应用框架对比

| 维度 | LangChain | LlamaIndex | Semantic Kernel | Haystack (deepset) |
| --- | --- | --- | --- | --- |
| **定位** | 通用 LLM 应用框架 | RAG 专用 | 企业级 LLM 框架 | NLP Pipeline 框架 |
| **主语言** | Python/JS | Python | C#/Python | Python |
| **学习曲线** | 中-高 | 低-中 | 中 | 中 |
| **RAG 能力** | 中（需组合） | 最强 | 中 | 强 |
| **Agent 能力** | 最强 | 基础 | 中 | 中 |
| **生态/集成** | 最丰富 | 丰富 | 微软生态 | 中等 |
| **状态管理** | LangGraph | 基础 | 内置 | Pipeline |
| **可观测性** | LangSmith | LlamaTrace | Azure Monitor | deepset Cloud |
| **稳定性** | 迭代快，API 变化多 | 较稳定 | 稳定 | 稳定 |

场景选型指南

| 需求场景 | 推荐框架 | 理由 |
| --- | --- | --- |
| 构建复杂 Agent 系统 | LangChain + LangGraph | Agent 和状态机能力最强 |
| 纯 RAG 应用 | LlamaIndex | RAG 专用，开箱即用 |
| .NET/企业级应用 | Semantic Kernel | 微软生态深度集成 |
| 生产级 NLP Pipeline | Haystack | Pipeline 抽象成熟 |
| 快速原型验证 | LangChain（LCEL） | 生态最丰富，组合灵活 |
| 轻量级简单场景 | 直接调用 API | 避免框架抽象的开销 |


<!-- Converted from: 02_LangChain与框架对比.html -->
