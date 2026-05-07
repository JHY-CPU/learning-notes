# 5_Agent 记忆系统

## 1. 记忆系统概览

Agent 的记忆系统是其区别于普通 Chatbot 的关键特征。借鉴认知科学中人类记忆的分类，Agent 记忆分为三个层次。

```
┌─────────────────────────────────────────────────┐
│              Agent 记忆系统架构                   │
│                                                  │
│  ┌─────────────┐                                │
│  │ 短期记忆     │ ← 当前会话上下文               │
│  │ (Working)   │   上下文窗口内                  │
│  └──────┬──────┘                                │
│         ↓ 总结/遗忘                              │
│  ┌─────────────┐                                │
│  │ 长期记忆     │ ← 跨会话持久化                  │
│  │ (Long-term) │   向量数据库存储                │
│  └──────┬──────┘                                │
│         ↓ 检索                                   │
│  ┌─────────────┐                                │
│  │ 工作记忆     │ ← 当前任务的临时状态             │
│  │ (Episodic)  │   任务完成后可归档               │
│  └─────────────┘                                │
└─────────────────────────────────────────────────┘
```

### 人类记忆 vs Agent 记忆

| 认知科学概念 | Agent 对应 | 实现方式 |
|------------|-----------|---------|
| 感觉记忆 | 原始输入 | 用户消息、工具返回值 |
| 工作记忆 | 上下文窗口 | Prompt 中的对话历史 |
| 短期记忆 | 会话摘要 | 压缩后的对话总结 |
| 长期记忆 | 向量数据库 | 嵌入 + 检索 |
| 情景记忆 | 任务执行日志 | 事件序列记录 |
| 语义记忆 | 知识图谱/文档库 | 结构化知识存储 |

## 2. 短期记忆（上下文窗口）

### 2.1 上下文窗口管理

```python
class ContextWindow:
    """管理 LLM 上下文窗口"""

    def __init__(self, max_tokens: int = 8192, reserve_output: int = 1024):
        self.max_tokens = max_tokens
        self.available_tokens = max_tokens - reserve_output
        self.messages: list[dict] = []

    def add_message(self, role: str, content: str) -> bool:
        """添加消息，如果超出窗口则进行压缩"""
        token_count = self.count_tokens(content)

        if self.current_tokens() + token_count > self.available_tokens:
            self.compress()

        if self.current_tokens() + token_count > self.available_tokens:
            self.evict_old_messages()

        self.messages.append({"role": role, "content": content})
        return True

    def current_tokens(self) -> int:
        return sum(self.count_tokens(m["content"]) for m in self.messages)

    def compress(self):
        """将旧消息压缩为摘要"""
        if len(self.messages) <= 4:
            return

        # 保留 system prompt 和最近的对话
        system_msgs = [m for m in self.messages if m["role"] == "system"]
        recent_msgs = self.messages[-4:]
        middle_msgs = self.messages[1:-4]

        if middle_msgs:
            summary = self.summarize(middle_msgs)
            self.messages = system_msgs + [
                {"role": "system", "content": f"先前对话摘要: {summary}"}
            ] + recent_msgs

    def evict_old_messages(self):
        """移除最旧的非 system 消息"""
        for i, msg in enumerate(self.messages):
            if msg["role"] != "system":
                self.messages.pop(i)
                break
```

### 2.2 消息压缩策略

```python
class MessageCompressor:
    """多种压缩策略"""

    def summarize(self, messages: list[dict]) -> str:
        """LLM 总结压缩"""
        conversation = "\n".join(
            f"{m['role']}: {m['content']}" for m in messages
        )
        return llm.generate(
            f"请总结以下对话的关键信息，保留重要决策和结论：\n{conversation}"
        )

    def extract_key_info(self, messages: list[dict]) -> dict:
        """提取结构化关键信息"""
        return {
            "entities": self.extract_entities(messages),
            "decisions": self.extract_decisions(messages),
            "pending_tasks": self.extract_tasks(messages),
            "user_preferences": self.extract_preferences(messages),
        }

    def sliding_window(self, messages: list[dict], window_size: int = 10) -> list[dict]:
        """滑动窗口：只保留最近 N 条"""
        return messages[-window_size:]
```

## 3. 长期记忆（向量数据库）

### 3.1 长期记忆存储

```python
from dataclasses import dataclass
from datetime import datetime
import hashlib

@dataclass
class Memory:
    content: str
    embedding: list[float]
    metadata: dict
    memory_type: str  # "fact", "preference", "event", "skill"
    importance: float  # 0-1
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0

class LongTermMemory:
    def __init__(self, embedding_model, vector_db):
        self.embed_model = embedding_model
        self.vdb = vector_db

    def store(self, content: str, memory_type: str = "fact",
              importance: float = 0.5, metadata: dict = None):
        """存储新记忆"""
        embedding = self.embed_model.encode(content)

        memory = Memory(
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )

        # 生成唯一 ID
        memory_id = hashlib.md5(content.encode()).hexdigest()
        self.vdb.upsert(memory_id, embedding, memory)

    def retrieve(self, query: str, top_k: int = 5,
                 memory_type: str = None, min_importance: float = 0.0) -> list[Memory]:
        """检索相关记忆"""
        query_embedding = self.embed_model.encode(query)

        results = self.vdb.search(
            query_embedding,
            top_k=top_k * 2,  # 多取一些再过滤
            filter={"memory_type": memory_type} if memory_type else None
        )

        filtered = []
        for memory, score in results:
            if memory.importance >= min_importance:
                memory.access_count += 1
                memory.last_accessed = datetime.now()
                filtered.append((memory, score))

        return filtered[:top_k]

    def forget(self, max_age_days: int = 90, min_importance: float = 0.1):
        """遗忘低重要性、长时间未访问的记忆"""
        cutoff = datetime.now() - timedelta(days=max_age_days)

        to_remove = []
        for mid, memory in self.vdb.get_all():
            if (memory.importance < min_importance and
                memory.last_accessed < cutoff and
                memory.access_count < 3):
                to_remove.append(mid)

        for mid in to_remove:
            self.vdb.delete(mid)
```

### 3.2 重要性评估

```python
class ImportanceEvaluator:
    """评估记忆的重要程度"""

    def evaluate(self, content: str, context: dict) -> float:
        score = 0.5  # 基础分

        # 1. 用户显式标记重要
        if any(kw in content for kw in ["记住", "重要", "不要忘记", "always"]):
            score += 0.3

        # 2. 包含个人信息
        if any(kw in content for kw in ["我叫", "我的", "我喜欢", "我不喜欢"]):
            score += 0.2

        # 3. 决策性内容
        if any(kw in content for kw in ["决定", "选择", "确定", "方案"]):
            score += 0.15

        # 4. 工具返回的关键结果
        if context.get("source") == "tool_result" and context.get("is_final"):
            score += 0.1

        # 5. 重复提及增加重要性
        repeat_count = context.get("repeat_count", 0)
        score += min(repeat_count * 0.05, 0.2)

        return min(score, 1.0)
```

## 4. 工作记忆（任务状态）

```python
from typing import Any

class WorkingMemory:
    """当前任务的工作记忆"""

    def __init__(self):
        self.goal: str = ""
        self.plan: list[str] = []
        self.current_step: int = 0
        self.scratchpad: dict[str, Any] = {}  # 临时变量
        self.observations: list[dict] = []     # 观察记录
        self.confidence: float = 1.0

    def set_goal(self, goal: str):
        self.goal = goal
        self.observations.append({
            "type": "goal_set",
            "content": goal,
            "time": datetime.now()
        })

    def update_scratchpad(self, key: str, value: Any):
        """更新临时变量"""
        self.scratchpad[key] = value

    def record_observation(self, observation: dict):
        """记录新的观察结果"""
        self.observations.append({
            **observation,
            "step": self.current_step,
            "time": datetime.now()
        })

    def advance_step(self):
        self.current_step += 1

    def get_state_summary(self) -> str:
        """获取当前状态摘要，用于注入 prompt"""
        return f"""
当前目标: {self.goal}
执行进度: 步骤 {self.current_step}/{len(self.plan)}
当前步骤: {self.plan[self.current_step] if self.plan else 'N/A'}
置信度: {self.confidence:.2f}
关键变量: {json.dumps(self.scratchpad, ensure_ascii=False, indent=2)}
"""

    def archive_to_long_term(self, long_term_memory: LongTermMemory):
        """任务完成后，将工作记忆归档到长期记忆"""
        summary = self.get_state_summary()
        long_term_memory.store(
            content=f"任务: {self.goal}\n结果: {summary}",
            memory_type="event",
            importance=0.7,
            metadata={"task_goal": self.goal}
        )
```

## 5. 记忆检索策略

### 5.1 混合检索

```python
class HybridMemoryRetriever:
    """结合向量相似度 + 时间衰减 + 重要性加权"""

    def __init__(self, vector_db, alpha=0.6, beta=0.2, gamma=0.2):
        self.vdb = vector_db
        self.alpha = alpha  # 语义相关度权重
        self.beta = beta    # 时间衰减权重
        self.gamma = gamma  # 重要性权重

    def retrieve(self, query: str, top_k: int = 5) -> list[Memory]:
        query_emb = self.embed(query)

        # 向量检索获取候选
        candidates = self.vdb.search(query_emb, top_k=top_k*3)

        scored = []
        for memory, semantic_score in candidates:
            # 时间衰减：越近的越好
            days_ago = (datetime.now() - memory.last_accessed).days
            time_score = 1.0 / (1.0 + days_ago * 0.1)

            # 综合评分
            final_score = (
                self.alpha * semantic_score +
                self.beta * time_score +
                self.gamma * memory.importance
            )
            scored.append((memory, final_score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]
```

### 5.2 记忆反思 (Memory Reflection)

```python
class MemoryReflection:
    """定期反思记忆，生成高层级洞察"""

    def reflect(self, memories: list[Memory], llm) -> list[str]:
        """从近期记忆中提取模式和洞察"""
        recent_content = "\n".join(
            f"- [{m.memory_type}] {m.content}"
            for m in memories[-20:]
        )

        insights = llm.generate(f"""
基于以下最近的记忆记录，找出：
1. 重复出现的主题或模式
2. 用户的偏好和习惯
3. 重要的结论或决策
4. 待解决的问题

记忆记录:
{recent_content}

输出洞察列表 (JSON 格式):
""")

        return json.loads(insights)
```

## 6. 完整记忆系统整合

```python
class AgentMemorySystem:
    """完整的三层记忆系统"""

    def __init__(self, config: dict):
        self.context = ContextWindow(max_tokens=config.get("max_tokens", 8192))
        self.long_term = LongTermMemory(
            embedding_model=config["embedding_model"],
            vector_db=config["vector_db"]
        )
        self.working = WorkingMemory()
        self.importance_eval = ImportanceEvaluator()
        self.retriever = HybridMemoryRetriever(config["vector_db"])

    def process_input(self, user_message: str) -> str:
        """处理用户输入，检索相关记忆"""
        # 1. 检索长期记忆
        relevant_memories = self.retriever.retrieve(user_message, top_k=3)

        # 2. 构建增强上下文
        memory_context = "\n".join(
            f"[记忆] {m.content}" for m in relevant_memories
        )

        # 3. 注入到上下文窗口
        enhanced_input = f"{memory_context}\n\n用户: {user_message}"
        self.context.add_message("user", enhanced_input)

        return enhanced_input

    def process_output(self, response: str):
        """处理 Agent 输出，决定是否存储"""
        self.context.add_message("assistant", response)

        # 评估重要性
        importance = self.importance_eval.evaluate(response, {})

        if importance > 0.6:
            self.long_term.store(
                content=response,
                memory_type="fact",
                importance=importance
            )
```

## 总结

记忆系统是 Agent 实现**持续学习和个性化**的基础。核心设计原则：**(1) 分层管理** -- 不同生命周期的信息用不同存储；**(2) 智能遗忘** -- 不是所有信息都值得永久保存；**(3) 混合检索** -- 语义+时间+重要性的综合评分。
