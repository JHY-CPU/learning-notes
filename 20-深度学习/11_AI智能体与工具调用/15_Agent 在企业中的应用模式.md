# 15_Agent 在企业中的应用模式

## 1. 企业 Agent 应用全景

```
┌─────────────────────────────────────────────────────────┐
│               企业 AI Agent 应用矩阵                     │
├────────────┬──────────────┬────────────┬────────────────┤
│   部门     │    Agent     │   核心能力  │   典型工具      │
├────────────┼──────────────┼────────────┼────────────────┤
│ 客服       │ 对话 Agent   │ 问答+工单   │ CRM/知识库      │
│ 研发       │ Code Agent   │ 代码+测试   │ IDE/Git/CI     │
│ 数据       │ 分析 Agent   │ SQL+可视化  │ 数据库/BI      │
│ 运营       │ 文档 Agent   │ 摘要+生成   │ 文档系统       │
│ 销售       │ 线索 Agent   │ 调研+跟进   │ CRM/邮件       │
│ 法务       │ 合同 Agent   │ 审核+对比   │ 文档/知识库    │
│ HR        │ 招聘 Agent   │ 筛选+排程   │ ATS/日历       │
└────────────┴──────────────┴────────────┴────────────────┘
```

## 2. 客服 Agent

### 2.1 架构设计

```python
class CustomerServiceAgent:
    """智能客服 Agent"""

    def __init__(self, llm, config: dict):
        self.llm = llm
        self.knowledge_base = config["knowledge_base"]
        self.ticket_system = config["ticket_system"]
        self.escalation_rules = config["escalation_rules"]

    def handle_inquiry(self, customer_message: str,
                       customer_context: dict) -> dict:
        # 1. 意图识别
        intent = self.classify_intent(customer_message)

        # 2. 根据意图路由
        match intent:
            case "faq":
                return self.answer_faq(customer_message)
            case "order_status":
                return self.check_order(customer_context["customer_id"])
            case "technical_issue":
                return self.handle_technical(customer_message, customer_context)
            case "complaint":
                return self.escalate_to_human(customer_message, "complaint")
            case _:
                return self.general_response(customer_message)

    def classify_intent(self, message: str) -> str:
        return self.llm.generate(f"""
将客户消息分类为以下意图之一：
[faq, order_status, technical_issue, complaint, feedback, general]

客户消息: {message}
意图:
""").strip()

    def answer_faq(self, question: str) -> dict:
        # RAG 检索知识库
        relevant_docs = self.knowledge_base.search(question, top_k=3)

        if not relevant_docs:
            return {"response": "我需要为您转接人工客服以获得更准确的帮助。",
                    "action": "escalate"}

        answer = self.llm.generate(f"""
基于以下知识库内容回答客户问题。如果找不到答案，说明无法回答。

知识库内容:
{relevant_docs}

客户问题: {question}

回答 (友好、专业、简洁):
""")

        return {"response": answer, "source": "knowledge_base"}

    def handle_technical(self, issue: str, context: dict) -> dict:
        """处理技术问题，必要时创建工单"""
        # 尝试自助解决
        solution = self.knowledge_base.search(f"故障排除 {issue}", top_k=3)

        if solution:
            return {"response": f"根据我们的知识库，建议您尝试：\n{solution}",
                    "action": "self_service"}

        # 创建工单
        ticket = self.ticket_system.create(
            customer_id=context["customer_id"],
            issue_type="technical",
            description=issue,
            priority=self.assess_priority(issue)
        )

        return {
            "response": f"已为您创建工单 #{ticket['id']}，技术团队将在24小时内回复您。",
            "action": "ticket_created",
            "ticket_id": ticket["id"]
        }
```

## 3. 数据分析 Agent

```python
class DataAnalysisAgent:
    """智能数据分析 Agent"""

    def __init__(self, llm, db_connection, chart_engine):
        self.llm = llm
        self.db = db_connection
        self.charts = chart_engine

    def analyze(self, question: str) -> dict:
        # 1. 理解分析需求
        analysis_plan = self.llm.generate(f"""
分析需求: {question}

生成分析计划 (JSON):
{{
  "data_queries": ["SQL查询1", "SQL查询2"],
  "analysis_type": "trend/comparison/distribution/correlation",
  "visualizations": ["chart_type1", "chart_type2"],
  "key_metrics": ["metric1", "metric2"]
}}
""")
        plan = json.loads(analysis_plan)

        # 2. 执行数据查询
        data_results = {}
        for query in plan["data_queries"]:
            # 安全执行 SQL（只读）
            data_results[query] = self.safe_sql_execute(query)

        # 3. 生成可视化
        charts = []
        for viz in plan["visualizations"]:
            chart = self.charts.create(data_results, viz)
            charts.append(chart)

        # 4. 生成分析报告
        report = self.llm.generate(f"""
基于以下数据结果生成分析报告：

原始问题: {question}
数据结果: {json.dumps(data_results, ensure_ascii=False, default=str)[:2000]}

报告应包括：
1. 核心发现 (3-5条)
2. 数据支撑
3. 趋势/异常说明
4. 建议行动
""")

        return {
            "report": report,
            "charts": charts,
            "data": data_results,
            "plan": plan
        }

    def safe_sql_execute(self, query: str) -> list[dict]:
        """安全执行只读 SQL"""
        # SQL 安全检查
        dangerous_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE"]
        query_upper = query.upper()
        for kw in dangerous_keywords:
            if kw in query_upper:
                raise ValueError(f"SQL 包含危险关键字: {kw}")

        # 添加 LIMIT 防止大量数据返回
        if "LIMIT" not in query_upper:
            query += " LIMIT 10000"

        return self.db.execute(query)
```

## 4. 文档处理 Agent

```python
class DocumentAgent:
    """文档处理 Agent"""

    def __init__(self, llm, doc_store):
        self.llm = llm
        self.doc_store = doc_store

    def summarize(self, document_path: str,
                  max_length: int = 500) -> str:
        """文档摘要"""
        content = self.doc_store.read(document_path)
        return self.llm.generate(f"""
请生成以下文档的摘要，不超过 {max_length} 字：

{content[:8000]}

摘要要求：
- 保留核心论点和关键数据
- 结构清晰
- 语言简洁
""")

    def compare_documents(self, doc_paths: list[str],
                          aspects: list[str]) -> dict:
        """多文档对比"""
        docs = {p: self.doc_store.read(p) for p in doc_paths}

        comparison = {}
        for aspect in aspects:
            comparison[aspect] = self.llm.generate(f"""
对比以下文档在"{aspect}"方面的异同：

{chr(10).join(f'文档 {i+1} ({path}): {content[:2000]}' for i, (path, content) in enumerate(docs.items()))}

输出结构化对比表。
""")
        return comparison

    def extract_info(self, document_path: str,
                     schema: dict) -> dict:
        """结构化信息提取"""
        content = self.doc_store.read(document_path)

        return json.loads(self.llm.generate(f"""
从以下文档中提取指定信息：

文档内容:
{content[:5000]}

目标 Schema:
{json.dumps(schema, ensure_ascii=False, indent=2)}

输出 JSON，字段缺失填 null。
"""))
```

## 5. 企业 Agent 部署架构

```
┌─────────────────────────────────────────────────────┐
│                  企业 Agent 平台架构                  │
├─────────────────────────────────────────────────────┤
│                   API Gateway                        │
│              (认证/限流/路由/日志)                    │
├──────────┬──────────┬──────────┬───────────────────┤
│ 客服Agent │ 分析Agent │ 文档Agent │ 自定义Agent       │
├──────────┴──────────┴──────────┴───────────────────┤
│                   Agent 运行时                       │
│          (沙箱/权限/监控/成本追踪)                   │
├─────────────────────────────────────────────────────┤
│                   共享服务层                         │
│   LLM 网关 │ 向量数据库 │ 工具市场 │ 知识库          │
├─────────────────────────────────────────────────────┤
│                   基础设施层                         │
│    K8s │ 日志系统 │ 监控告警 │ CI/CD               │
└─────────────────────────────────────────────────────┘
```

## 6. 成本控制

```python
class CostController:
    """Agent 成本控制"""

    def __init__(self, budget_config: dict):
        self.daily_budget = budget_config.get("daily_usd", 100)
        self.per_request_budget = budget_config.get("per_request_usd", 1.0)
        self.spent_today = 0

    def check_budget(self, estimated_cost: float) -> tuple[bool, str]:
        if self.spent_today + estimated_cost > self.daily_budget:
            return False, "日预算即将耗尽"
        if estimated_cost > self.per_request_budget:
            return False, "单次请求超预算"
        return True, "OK"

    def optimize_model_selection(self, task_complexity: str) -> str:
        """根据任务复杂度选择模型"""
        model_tiers = {
            "simple": {"model": "gpt-4o-mini", "cost_per_1k": 0.00015},
            "medium": {"model": "gpt-4o", "cost_per_1k": 0.005},
            "complex": {"model": "gpt-4", "cost_per_1k": 0.03},
        }
        return model_tiers[task_complexity]["model"]
```

## 7. 落地建议

```
企业 Agent 落地路线图：

Phase 1 (1-2月): 试点
  - 选择高频、低风险场景（如 FAQ 客服）
  - 最小可行产品：RAG + 单 Agent
  - 关键指标：准确率 > 85%，人工接管率 < 20%

Phase 2 (2-4月): 扩展
  - 增加工具和数据源
  - 引入多步推理和自我反思
  - 关键指标：解决率 > 70%，满意度 > 4.0

Phase 3 (4-6月): 深化
  - Multi-Agent 协作
  - 跨系统集成
  - 关键指标：自动化率 > 50%，ROI > 3x

Phase 4 (持续): 优化
  - 持续评估和改进
  - 安全审计
  - 成本优化
```

## 总结

企业 Agent 应用的核心在于**将 LLM 能力与业务系统深度集成**。成功的关键因素：**(1) 选对场景** -- 高频、重复、有明确输入输出的任务；**(2) 稳健的工具集成** -- 安全、可靠的系统对接；**(3) 渐进式落地** -- 从试点到扩展到深化。
