# LLM Agent架构 - Agent与工具调用

*深入理解 ReAct 模式、工具调用（Tool/Function Calling）、规划策略（CoT/ToT）、记忆机制及多 Agent 协作架构*

ChatBot vs Agent vs Agentic Workflow

| 类型 | 特点 | 复杂度 | 示例 |
| --- | --- | --- | --- |
| 简单 ChatBot | 单轮/多轮对话，无工具 | 低 | 客服机器人 |
| 单 Agent | 自主决策，使用工具完成任务 | 中 | AI 编程助手、数据分析助手 |
| 多 Agent 协作 | 多个专业 Agent 分工协作 | 高 | 软件开发团队模拟、复杂研究 |
| Agentic Workflow | 结构化流程中嵌入 Agent 节点 | 中-高 | 自动化审批流程、CI/CD 管线 |

ReAct vs Function Calling 对比

| 维度 | ReAct（文本解析） | Function Calling（结构化） |
| --- | --- | --- |
| 格式 | 自由文本中的 Action/Action Input | 结构化 JSON function_call |
| 可靠性 | 可能格式解析失败 | 结构化输出，极可靠 |
| 并行调用 | 串行 | 支持并行调用多个工具 |
| 强制调用 | 无法强制 | 可强制指定工具调用 |
| 支持模型 | 所有 LLM | GPT-4/Claude/Gemini 等 |

Agent 记忆类型

| 类型 | 存储位置 | 生命周期 | 实现方式 |
| --- | --- | --- | --- |
| **短期记忆** | 对话上下文 | 当前会话 | 消息历史列表，受上下文窗口限制 |
| **长期记忆** | 外部存储 | 持久化 | 向量数据库 / KV 存储 / 知识图谱 |
| **工作记忆** | Agent 内部状态 | 当前任务 | 中间变量、工具执行结果 |
| **情景记忆** | 历史经验 | 持久化 | 任务执行记录、成功/失败案例 |
| **语义记忆** | 知识库 | 持久化 | 事实、规则、领域知识 |

多 Agent 协作架构模式

| 模式 | 说明 | 适用场景 |
| --- | --- | --- |
| **链式（Sequential）** | Agent 按固定顺序依次执行 | 数据处理管线、审批流程 |
| **路由（Routing）** | Router Agent 将请求分发给专业 Agent | 客服系统、多领域问答 |
| **层级（Hierarchical）** | 管理者 Agent 分配任务给下属 Agent | 项目管理、复杂任务分解 |
| **辩论（Debate）** | 多个 Agent 互相辩论以提升质量 | 事实核查、决策优化 |
| **群聊（Group Chat）** | Agent 在共享频道中自由讨论 | 头脑风暴、团队模拟 |


<!-- Converted from: 01_LLM Agent架构.html -->
