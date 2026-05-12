# AI智能体与工具调用

## 一、AI Agent概述

### 1.1 定义

AI Agent（智能体）是能够感知环境、做出决策并采取行动的AI系统。

核心能力：
- **规划**：将复杂任务分解为子任务
- **记忆**：短期（上下文）和长期（外部存储）记忆
- **工具使用**：调用外部API和工具
- **反思**：评估和改进自身行为

### 1.2 Agent架构

```
用户指令 → 规划器 → 执行器 → 工具调用
    ↑                          ↓
    └──── 反思/修正 ←──────────┘
```

---

## 二、ReAct框架

Reasoning + Acting：交替进行推理和行动。

```
Thought: 我需要查询今天的天气
Action: search("今天北京天气")
Observation: 晴，最高温28度
Thought: 用户需要穿衣建议
Final Answer: 今天北京晴天，气温28度，建议穿短袖。
```

---

## 三、工具调用

### 3.1 常见工具类型

- 搜索引擎
- 代码执行器
- 数据库查询
- API调用（天气、地图、金融等）
- 文件读写
- 网页浏览

### 3.2 Function Calling

大模型原生支持结构化工具调用：
```json
{
  "name": "get_weather",
  "arguments": {"city": "Beijing", "date": "2024-01-01"}
}
```

---

## 四、代表系统

- **AutoGPT**：自主任务执行
- **LangChain Agents**：工具编排框架
- **OpenAI Assistants API**：有状态的Agent
- **Claude Tool Use**：结构化工具调用
- **MetaGPT**：多Agent协作

---

## 五、多Agent协作

- 角色扮演：不同Agent扮演不同角色
- 讨论/辩论：多Agent交换意见
- 层级结构：管理者Agent调度执行Agent
- 代表：ChatDev、AutoGen、CrewAI
