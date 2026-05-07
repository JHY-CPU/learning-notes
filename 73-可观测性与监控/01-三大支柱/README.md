# 01-三大支柱

## 可观测性 vs 监控

- **监控（Monitoring）**：基于已知的预定义指标和阈值，回答"系统是否正常"。关注已知的故障模式。
- **可观测性（Observability）**：从系统的外部输出（指标、日志、追踪）推断内部状态，回答"为什么出问题了"。能够探索未知的故障模式。

监控是可观测性的子集。可观测性的三个核心信号（Metrics、Logs、Traces）构成了"三大支柱"。

---

## 三大支柱概述

| 支柱 | 数据类型 | 核心价值 | 代表工具 |
|------|----------|----------|----------|
| **Metrics** | 数值型时间序列 | 趋势分析、告警、容量规划 | Prometheus、InfluxDB |
| **Logging** | 离散事件文本 | 故障排查、审计、调试 | ELK Stack、Loki |
| **Tracing** | 请求调用链 | 分布式追踪、瓶颈定位 | Jaeger、Zipkin |

---

## Metrics（指标）

### 指标类型

| 类型 | 含义 | 典型场景 |
|------|------|----------|
| **Counter** | 只增不减的计数器 | 请求数、错误数、处理的消息数 |
| **Gauge** | 可增可减的瞬时值 | CPU 利用率、队列长度、温度 |
| **Histogram** | 对观测值分桶统计 | 请求延迟分布（自动计算分位数） |
| **Summary** | 客户端计算分位数 | 精确的 P50/P95/P99（不支持聚合） |

### Prometheus 数据模型

- 每个指标由 **指标名** 和一组 **标签（Label）** 唯一标识。
- 示例：`http_requests_total{method="POST", handler="/api/users", status="200"} 1027`
- 四种数据类型对应四种 metric type：counter、gauge、histogram、summary。
- 通过 HTTP pull 模型从 target 采集数据（也可通过 Pushgateway 推送）。

### PromQL 查询语言

#### 选择器与函数

```promql
# 即时向量选择
http_requests_total{method="POST", status=~"5.."}

# 范围向量选择（过去5分钟）
http_requests_total{job="api"}[5m]

# 常用函数
rate(http_requests_total[5m])            # 每秒增长率（Counter 适用）
irate(http_requests_total[5m])           # 瞬时增长率
increase(http_requests_total[1h])        # 1小时内的增量
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))  # P99 延迟
```

#### 聚合操作

```promql
sum(rate(http_requests_total[5m])) by (handler)       # 按 handler 聚合 QPS
avg(node_memory_MemAvailable_bytes) by (instance)     # 按实例聚合平均可用内存
topk(3, rate(http_requests_total[5m]))                # QPS 最高的 3 个指标
count(up == 1) by (job)                               # 每个 job 的健康实例数
```

#### 常用查询示例

```promql
# 请求错误率
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# 请求延迟 P99
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# CPU 使用率
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 磁盘使用率
(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100
```

### Prometheus 架构

- **Prometheus Server**：核心组件，负责指标采集、存储和查询。
- **TSDB**：本地时序数据库，数据保留期可配置，默认 15 天。
- **Exporters**：将第三方系统指标转换为 Prometheus 格式（node_exporter、mysqld_exporter 等）。
- **Pushgateway**：短生命周期 Job 推送指标的中转站。
- **Alertmanager**：处理告警通知（路由、分组、抑制、静默）。
- **Grafana**：可视化展示，通过 PromQL 查询 Prometheus 数据。

### 服务发现与 Target 管理

- 支持多种服务发现：Kubernetes、Consul、EC2、Azure、文件（file_sd）。
- 静态配置：`static_configs` 直接指定 target 地址。
- Relabeling：在采集前/后对标签进行转换、过滤、重命名。

### 告警规则（Alertmanager）

```yaml
groups:
  - name: example
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.01
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

Alertmanager 特性：告警分组（group_by）、抑制（inhibit）、静默（silence）、路由（route）。

### 长期存储方案

| 方案 | 特点 |
|------|------|
| **Thanos** | CNCF 项目，添加全局查询、跨集群聚合、无限存储到 S3 |
| **Grafana Mimir** | Grafana 出品，多租户、水平扩展、兼容 PromQL |
| **VictoriaMetrics** | 高性能时序数据库，比 Prometheus 节省 10 倍存储，兼容 Prometheus 协议 |

---

## Logging（日志）

### 日志级别

| 级别 | 用途 |
|------|------|
| **DEBUG** | 详细的调试信息，仅开发/排查时开启 |
| **INFO** | 记录关键业务流程（请求进入、订单创建、支付完成） |
| **WARN** | 潜在问题（重试、降级、配置不推荐） |
| **ERROR** | 业务异常、系统错误，需要关注但不影响整体运行 |
| **FATAL** | 致命错误，进程无法继续运行 |

### 结构化日志（JSON 格式）

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "ERROR",
  "logger": "com.example.OrderService",
  "message": "Failed to process order",
  "traceId": "abc123",
  "spanId": "def456",
  "orderId": "ORD-001",
  "error": "ConnectionTimeoutException",
  "stackTrace": "..."
}
```

结构化日志便于机器解析、索引和查询，比纯文本日志更适合生产环境。

### ELK/EFK Stack

- **Elasticsearch**：分布式搜索和分析引擎，存储和索引日志。
- **Logstash / Fluentd**：日志采集和处理管道。Fluentd 比 Logstash 更轻量，适合容器环境。
- **Kibana**：日志可视化和查询界面，支持 Lucene/KQL 语法。

架构：应用 -> Logstash/Fluentd -> Elasticsearch -> Kibana。

### Loki（轻量级日志聚合）

- Grafana Labs 出品，"像 Prometheus 但用于日志"。
- 只索引标签（metadata），不索引日志全文，存储成本远低于 Elasticsearch。
- 使用 LogQL 查询语言，与 PromQL 风格一致。
- 适合与 Prometheus/Grafana 搭配使用，保持统一的工具链。

### 日志采样与降级

- **采样**：高流量场景下对 DEBUG/INFO 日志采样（如每 100 条记录 1 条），减少存储和网络开销。
- **降级**：系统资源紧张时自动切换到更低的日志级别。
- **动态调整**：通过 API 或配置中心在线调整日志级别，无需重启服务。

### 日志最佳实践

- 使用结构化日志（JSON），包含 traceId 实现日志与链路关联。
- 日志内容包含上下文信息（userId、orderId、requestId），便于关联分析。
- 避免在循环中打印日志、避免打印敏感信息（密码、Token）。
- 生产环境默认 INFO 级别，按需动态调整为 DEBUG。
- 设置合理的日志保留策略（7-30 天），避免磁盘占满。

---

## Tracing（链路追踪）

### 分布式追踪概念

- **Trace**：一次完整的请求链路，由多个 Span 组成。
- **Span**：链路中的一个工作单元（如一次 RPC 调用、一次数据库查询）。
- **TraceID**：全局唯一标识一次请求链路，随请求在服务间传播。
- **SpanID**：标识链路中的一个 Span。
- **ParentSpanID**：标识 Span 的父节点，构建调用树。
- **Context Propagation**：通过 HTTP Header（如 `X-B3-TraceId`、`traceparent`）在服务间传递 Trace 上下文。

### OpenTelemetry 标准

OpenTelemetry（OTel）是 CNCF 旗下统一的可观测性标准，整合了 OpenTracing 和 OpenCensus。

核心组件：
- **OTel API**：定义追踪、指标、日志的接口。
- **OTel SDK**：API 的实现，支持配置采样、导出器、资源属性。
- **OTel Collector**：接收、处理、导出遥测数据的中间件，支持多种数据源和后端。
- **OTLP**：OpenTelemetry Protocol，统一的数据传输协议（gRPC/HTTP）。

### SDK 集成与自动埋点

- **Java**：`opentelemetry-javaagent.jar` 通过 Java Agent 自动注入，支持 Spring Boot、JDBC、HTTP Client 等常用框架。
- **Go**：使用 `otelhttp`、`otelgrpc` 中间件自动追踪 HTTP/gRPC 调用。
- **Python**：`opentelemetry-instrument` 自动注入，或手动使用装饰器。
- **Node.js**：`@opentelemetry/sdk-trace-node` 自动 patch 常用模块（Express、HTTP、pg 等）。

### Jaeger / Zipkin

| 特性 | Jaeger | Zipkin |
|------|--------|--------|
| 开发者 | Uber | Twitter |
| 存储 | Cassandra、Elasticsearch | Cassandra、Elasticsearch、MySQL |
| UI | 功能丰富，支持 Trace 对比 | 简洁实用 |
| 兼容性 | OpenTracing/OTel | OpenTracing/B3 |
| 特色 | 依赖图、性能对比、自适应采样 | 轻量、部署简单 |

### 采样策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| **固定比率** | 按固定百分比采样（如 10%） | 流量稳定、成本敏感 |
| **速率限制** | 限制每秒采样的 Trace 数量 | 高流量场景 |
| **Tail 采样** | 先收集完整 Trace，再根据结果决定是否保留（如只保留错误 Trace） | 需要捕获异常但不想采样所有请求 |

### 链路分析与瓶颈定位

- **瀑布图（Waterfall）**：展示 Span 的时间线，直观定位最慢的调用。
- **关键路径分析**：找到 Trace 中决定总耗时的最长路径。
- **Span 统计**：按服务/操作聚合 Span 的延迟、错误率。
- **对比分析**：对比快/慢 Trace 的差异，定位性能退化原因。
- 与 Metrics/Logs 关联：从 Trace 跳转到相关日志，从告警跳转到对应 Trace。
