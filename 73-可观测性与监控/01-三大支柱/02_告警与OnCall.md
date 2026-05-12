# 告警与OnCall - 可观测性与监控笔记


## 告警与 OnCall


## 一、告警设计原则


### 1. 好告警的标准


- **可操作性（Actionable）：**
   每条告警都应该对应一个明确的操作
- **高信噪比：**
   真正的问题告警 > 噪声告警
- **上下文丰富：**
   包含排查所需的基本信息
- **分级明确：**
   不同严重程度的告警走不同通道


### 2. 告警信号分类


| 级别 | 信号 | 示例 | 响应要求 |
| --- | --- | --- | --- |
| P0 / Critical | 用户受损，服务不可用 | 核心 API 5xx > 5% | 立即响应（<5min） |
| P1 / High | 用户受损，部分降级 | 延迟 P99 > 3s | 30min 内响应 |
| P2 / Medium | 即将影响用户 | 磁盘使用 > 85% | 工作时间处理 |
| P3 / Low | 技术债务/异常 | 证书 30 天内过期 | 下一迭代处理 |


## 二、多窗口告警（Multi-Window Alerts）


### 1. 问题：单窗口告警的缺陷


```
// 单窗口告警：error_rate > 5% for 5min
问题：
- 短暂毛刺触发误报（5% 持续 10 秒后恢复）
- 持续恶化可能被慢窗口掩盖

毛刺:    ████ 5.2% → 立刻恢复
真实故障: ████████████████ 持续 > 5%
```


### 2. 解决：双窗口 / 多窗口告警


```
// 双窗口告警：短期确认 + 长期趋势
规则：
  (error_rate > 5% FOR 2min)     ← 短窗口：确认当前有问题
  AND
  (error_rate > 2% FOR 15min)    ← 长窗口：确认不是瞬时毛刺

// 三窗口告警（Google SRE 推荐）
P0 告警：
  快速窗口：error > 10% for 2min   → 确认问题存在
  中速窗口：error > 5%  for 30min  → 确认问题持续
  慢速窗口：error > 2%  for 2h     → 确认是系统性问题

效果：过滤掉 90%+ 的毛刺告警，同时不漏掉真实故障
```


### 3. Prometheus 示例


```
# 单窗口（不推荐）
- alert: HighErrorRate
  expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
  for: 1m

# 双窗口（推荐）
- alert: HighErrorRate
  expr: |
    (rate(http_requests_total{status=~"5.."}[5m]) > 0.05)
    AND
    (rate(http_requests_total{status=~"5.."}[1h]) > 0.02)
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "错误率持续偏高"
    runbook: "https://wiki.internal/runbook/high-error-rate"
```


## 三、告警降噪策略


### 1. 常见噪声来源


- **阈值过于敏感：**
   正常波动触发告警
- **依赖告警风暴：**
   底层故障引发上层所有服务告警
- **重复告警：**
   同一问题持续触发
- **不可操作告警：**
   没有明确修复动作的告警


### 2. 降噪方法


```
# 1. 聚合（Grouping）
# 相同服务的多条告警合并为一条
group_by: [alertname, service, cluster]
group_wait: 30s        # 等待30s聚合同组告警
group_interval: 5min   # 同组新告警的最小间隔

# 2. 抑制（Inhibition）
# 如果高级别告警已触发，抑制低级别告警
inhibit_rules:
  - source_match: { severity: 'critical' }
    target_match: { severity: 'warning' }
    equal: [service]  # 同一服务

# 3. 静默（Silence）
# 已知维护期间，手动设置静默规则
# 例：数据库升级期间，抑制所有数据库相关告警

# 4. 节流（Throttling）
# 同一告警在 N 分钟内只通知一次
repeat_interval: 4h  # 同一告警每 4h 重复通知

# 5. 基于 SLO 的告警（推荐）
# 而非固定阈值
- alert: SLOBurnRateHigh
  expr: |
    (1 - slo_success_rate_5m) / (1 - slo_target) > 14.4
    # 14.4 倍 burn rate = 1h 内消耗 5% 的 error budget
```


## 四、On-Call 体系


### 1. On-Call 职责


- 响应和处理生产告警
- 故障排查和恢复
- 协调跨团队沟通
- 编写事后复盘（Post-mortem）
- 处理用户升级工单


### 2. 轮值（Rotation）策略


```
推荐轮值方案：
┌─────────────────────────────────────┐
│  主 On-Call（Primary）               │  ← 第一响应人
├─────────────────────────────────────┤
│  副 On-Call（Secondary）             │  ← 主 On-Call 无响应时升级
├─────────────────────────────────────┤
│  管理者（Manager）                   │  ← 严重故障时升级到管理层
└─────────────────────────────────────┘

轮值周期：1 周（推荐，太短交接频繁，太长压力大）
交接：周五下午交接会议，review 本周事件和待办
日历：On-Call 日历同步到团队日历，提前安排
```


### 3. 升级策略（Escalation Policy）


```
告警触发
  │
  ├── T+0min:   通知主 On-Call（电话 + 短信 + Slack）
  │
  ├── T+5min:   未确认 → 二次通知 + 通知副 On-Call
  │
  ├── T+15min:  未确认 → 通知团队 Lead
  │
  └── T+30min:  未确认 → 通知部门经理 + 启动 Incident 流程

PagerDuty / OpsGenie 配置：
escalation_rules:
  - delay_minutes: 5
    targets: [{ type: user, id: primary_oncall }]
  - delay_minutes: 10
    targets: [{ type: user, id: secondary_oncall }]
  - delay_minutes: 15
    targets: [{ type: user, id: team_lead }]
```


## 五、告警疲劳（Alert Fatigue）


> **Warning:** **危害：**
> 当告警噪声过高时，工程师开始忽略告警，导致真正的故障被遗漏。研究表明，在医疗领域，告警疲劳是导致事故的重要原因，在 DevOps 中同样致命。


### 1. 度量告警质量


```
关键指标：
- 信噪比 = 可操作告警数 / 总告警数（目标 > 50%）
- 平均响应时间（MTTA）
- 告警-事件关联率 = 触发告警中有对应故障的比例
- 误报率 = 无需操作的告警 / 总告警数（目标 < 30%）

Google 推荐：
- 每个团队每天不超过 6 条告警
- P0/P1 告警每月不超过 2 条
- 告警准确率 > 80%
```


### 2. 治理流程


```
1. 定期审计（每月）→ 审查所有告警规则
2. 告警分类 → 可操作 / 噪声 / 待优化
3. 删除或合并噪声告警
4. 将固定阈值改为 SLO-based 告警
5. 添加 runbook 链接（每个告警必须有）
6. 跟踪改进效果（信噪比变化）
```


## 六、Runbook（运行手册）


```
Runbook 模板：

# 告警: [告警名称]
## 症状
[告警的触发条件和表现]

## 影响范围
[影响哪些用户/服务]

## 排查步骤
1. 检查 [仪表盘链接]
2. 运行 [诊断命令]
3. 检查 [依赖服务状态]

## 恢复步骤
1. [具体操作]
2. [回滚命令（如需要）]

## 根因分析
[常见原因列表]

## 升级条件
如果以上步骤无效，升级到 [团队/联系人]
```


> **Note:** **面试要点：**
> 好告警三要素：可操作、高信噪比、上下文丰富。多窗口告警（短期确认+长期趋势）是 Google 推荐的降噪方法。On-Call 体系包含主/副轮值和分级升级策略。告警治理的核心指标是信噪比（目标>50%）。SLO-based 告警（burn rate）比固定阈值更科学。

总结：告警设计核心原则是可操作性和高信噪比；多窗口告警（双窗口/三窗口）有效过滤毛刺；降噪方法包括聚合、抑制、静默、节流和SLO-based告警；On-Call体系含主副轮值+分级升级策略；告警疲劳是严重问题需定期审计治理；每个告警必须附带Runbook


<!-- Converted from: 02_告警与OnCall.html -->
