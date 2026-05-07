# Prometheus监控

## 一、概念说明

Prometheus是开源的监控和告警系统，采用拉取（Pull）模式采集指标。通过PromQL查询语言实现灵活的数据分析。

| 概念 | 说明 |
|------|------|
| Metric | 时间序列指标 |
| Target | 被监控的目标 |
| Job | 一组相同类型的Target |
| Scrape | 采集指标数据 |
| Alert Rule | 告警规则 |

| 指标类型 | 说明 |
|----------|------|
| Counter | 只增计数器 |
| Gauge | 可增减的值 |
| Histogram | 数据分布（分桶） |
| Summary | 数据分布（分位数） |

## 二、具体用法

### 安装和配置

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  # Prometheus自身监控
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
        labels:
          env: 'production'

  # 应用指标
  - job_name: 'my-app'
    metrics_path: '/metrics'
    static_configs:
      - targets: ['app:8080']
    scrape_interval: 10s
```

```bash
# Docker方式运行
docker run -d \
    --name prometheus \
    -p 9090:9090 \
    -v /etc/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus:latest
```

### 应用埋点

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# 定义指标
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1, 5]
)

ACTIVE_REQUESTS = Gauge(
    'http_active_requests',
    'Number of active HTTP requests'
)

# 使用指标
@app.route('/api/data')
def get_data():
    ACTIVE_REQUESTS.inc()
    start = time.time()
    try:
        result = fetch_data()
        REQUEST_COUNT.labels('GET', '/api/data', '200').inc()
        return result
    except Exception:
        REQUEST_COUNT.labels('GET', '/api/data', '500').inc()
        raise
    finally:
        duration = time.time() - start
        REQUEST_LATENCY.labels('GET', '/api/data').observe(duration)
        ACTIVE_REQUESTS.dec()

# 启动metrics端口
start_http_server(8080)
```

```java
// Java Micrometer集成
@RestController
public class ApiController {
    private final Counter requestCounter;
    private final Timer requestTimer;

    public ApiController(MeterRegistry registry) {
        requestCounter = Counter.builder("http.requests.total")
            .tag("controller", "api")
            .register(registry);
        requestTimer = Timer.builder("http.request.duration")
            .publishPercentiles(0.5, 0.95, 0.99)
            .register(registry);
    }

    @GetMapping("/data")
    public Response getData() {
        requestCounter.increment();
        return requestTimer.record(() -> {
            return service.getData();
        });
    }
}
```

### PromQL查询

```bash
# 基本查询
http_requests_total

# 过滤
http_requests_total{method="GET", status="200"}

# 计算QPS（每秒请求数）
rate(http_requests_total[5m])

# 按标签聚合
sum(rate(http_requests_total[5m])) by (endpoint)

# 错误率
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m]))

# P99延迟
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

# CPU使用率
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 内存使用率
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100
```

### 告警规则

```yaml
# alert_rules.yml
groups:
  - name: application
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "高错误率告警"
          description: "错误率超过5%，当前值: {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P95延迟过高"
          description: "P95延迟超过1秒，当前值: {{ $value }}s"
```

## 三、注意事项与常见陷阱

1. **标签基数**：避免使用用户ID等高基数标签
2. **数据保留**：默认15天，长期存储考虑Thanos或VictoriaMetrics
3. **采集间隔**：根据需求设置，过短增加负载，过长遗漏问题
4. **Recording Rules**：复杂查询使用预计算减少查询压力
5. **联邦集群**：大规模部署使用联邦Prometheus
6. **磁盘空间**：监控Prometheus自身磁盘使用
7. **高可用**：部署多个Prometheus实例确保监控可用性
