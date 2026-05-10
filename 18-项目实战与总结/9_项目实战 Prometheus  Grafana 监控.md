# 项目实战 Prometheus + Grafana 监控


## 📦 项目实战 10: Prometheus + Grafana 监控


应用 Metrics 埋点 (请求计数/延迟分位/错误率)、Prometheus 配置与 targets、Grafana 仪表盘设计、Alertmanager 告警、Jaeger 分布式追踪。


## 应用 Metrics 埋点


```
// ========== Go Prometheus Metrics ==========
// 使用 prometheus/client_golang

var (
    // HTTP 请求计数 (method, path, status)
    httpRequestsTotal = prometheus.NewCounterVec(
        prometheus.CounterOpts{
            Name: "http_requests_total",
            Help: "Total number of HTTP requests",
        },
        []string{"method", "path", "status"},
    )

    // HTTP 请求延迟 (分位)
    httpRequestDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "http_request_duration_seconds",
            Help:    "HTTP request latency in seconds",
            Buckets: prometheus.DefBuckets,
            // DefBuckets: [.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]
        },
        []string{"method", "path"},
    )

    // 正在处理的请求数
    httpInFlight = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "http_requests_in_flight",
            Help: "Current number of HTTP requests in flight",
        },
    )

    // 业务指标: 数据库查询延迟
    dbQueryDuration = prometheus.NewHistogramVec(
        prometheus.HistogramOpts{
            Name:    "db_query_duration_seconds",
            Help:    "Database query latency",
            Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1},
        },
        []string{"query_type"},
    )

    // 业务指标: 活跃用户数
    activeUsers = prometheus.NewGauge(
        prometheus.GaugeOpts{
            Name: "active_users_total",
            Help: "Current active users",
        },
    )
)

func init() {
    prometheus.MustRegister(
        httpRequestsTotal, httpRequestDuration,
        httpInFlight, dbQueryDuration, activeUsers,
    )
}

// ========== Prometheus 中间件 ==========
func PrometheusMiddleware() gin.HandlerFunc {
    return func(c *gin.Context) {
        // 正在处理 +1
        httpInFlight.Inc()

        // 开始计时
        start := time.Now()

        // 记录响应状态码
        c.Next()

        // 延迟
        duration := time.Since(start).Seconds()
        path := c.FullPath()
        if path == "" {
            path = "unknown"
        }

        httpRequestDuration.WithLabelValues(
            c.Request.Method, path,
        ).Observe(duration)

        httpRequestsTotal.WithLabelValues(
            c.Request.Method, path,
            fmt.Sprintf("%d", c.Writer.Status()),
        ).Inc()

        // 正在处理 -1
        httpInFlight.Dec()
    }
}

// 暴露 /metrics 端点
// r.GET("/metrics", gin.WrapH(promhttp.Handler()))
```


## Prometheus 配置


```
# ========== prometheus.yml ==========
global:
  scrape_interval: 15s      # 采集间隔
  evaluation_interval: 15s  # 规则评估间隔

# Alertmanager 配置
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# 告警规则
rule_files:
  - "alerts/*.yml"

# 采集目标
scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'api-service'
    metrics_path: '/metrics'
    static_configs:
      - targets:
        - 'api:8080'
          # docker-compose 服务名
      - targets: ['host.docker.internal:8080']  # 本地

  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

# ========== 告警规则 ==========
# alerts/service.yml
groups:
  - name: service
    rules:
      - alert: HighErrorRate
        expr: |
          rate(http_requests_total{status=~"5.."}[5m])
          /
          rate(http_requests_total[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "API error rate > 5%"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: |
          histogram_quantile(0.99,
            rate(http_request_duration_seconds_bucket[5m])
          ) > 1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "P99 latency > 1s"

      - alert: InstanceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Instance {{ $labels.instance }} down"

      - alert: HighCPU
        expr: node_load1 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CPU load > 2 for 5 minutes"
```


## Grafana 仪表盘与 Jaeger


```
// ========== Grafana 仪表盘设计 ==========
// RED 指标 (微服务黄金信号):
// Rate:    请求率 (req/s)
// Error:   错误率 (%)
// Duration:延迟 (P50/P90/P99)

// 常用面板:
// 1. 请求率 — 按 method/path/status 分组
//    rate(http_requests_total[5m])
//
// 2. 错误率 — 5xx / 总请求
//    sum(rate(http_requests_total{status=~"5.."}[5m]))
//    / sum(rate(http_requests_total[5m]))
//
// 3. P99 延迟
//    histogram_quantile(0.99,
//      rate(http_request_duration_seconds_bucket[5m]))
//
// 4. CPU / 内存 (Node Exporter)
//    100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
//    node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes
//
// 5. 业务指标
//    active_users_total (Gauge)

// ========== Jaeger 分布式追踪 ==========
// 使用 OpenTelemetry SDK

// import (
//     "go.opentelemetry.io/otel"
//     "go.opentelemetry.io/otel/attribute"
//     "go.opentelemetry.io/otel/exporters/jaeger"
//     "go.opentelemetry.io/otel/sdk/trace"
// )

// 初始化 TracerProvider:
// func initTracer() (*trace.TracerProvider, error) {
//     exp, err := jaeger.New(
//         jaeger.WithCollectorEndpoint(
//             jaeger.WithEndpoint("http://jaeger:14268/api/traces"),
//         ),
//     )
//
//     tp := trace.NewTracerProvider(
//         trace.WithBatcher(exp),
//         trace.WithResource(resource.NewWithAttributes(
//             semconv.SchemaURL,
//             semconv.ServiceNameKey.String("api-service"),
//         )),
//     )
//     otel.SetTracerProvider(tp)
//     return tp, nil
// }

// 在请求中创建 Span:
// tracer := otel.Tracer("api-service")
// ctx, span := tracer.Start(ctx, "process_order")
// defer span.End()
//
// span.SetAttributes(
//     attribute.String("order.id", orderID),
//     attribute.Float64("amount", amount),
// )
//
// span.AddEvent("payment_processed", trace.WithAttributes(
//     attribute.String("status", "success"),
// ))

// ========== Docker Compose ==========
// docker-compose.yml:
// version: '3.8'
// services:
//   api:
//     build: .
//     ports: ["8080:8080"]
//
//   prometheus:
//     image: prom/prometheus
//     volumes:
//       - ./prometheus.yml:/etc/prometheus/prometheus.yml
//       - promdata:/prometheus
//     ports: ["9090:9090"]
//
//   grafana:
//     image: grafana/grafana
//     environment:
//       GF_SECURITY_ADMIN_PASSWORD: admin
//     volumes:
//       - grafana-data:/var/lib/grafana
//     ports: ["3000:3000"]
//
//   alertmanager:
//     image: prom/alertmanager
//     volumes:
//       - ./alertmanager.yml:/etc/alertmanager/alertmanager.yml
//
//   jaeger:
//     image: jaegertracing/all-in-one:1.57
//     ports:
//       - "16686:16686"  # UI
//       - "14268:14268"  # HTTP collector
//
//   node-exporter:
//     image: prom/node-exporter
//
// volumes:
//   promdata:
//   grafana-data:
```


> **Note:** 💡 Prometheus + Grafana 项目要点: Counter 累计计数 (请求数); Histogram 分位分布 (延迟); Gauge 当前值 (活跃连接); /metrics 端点暴露; PromQL 查询 (rate/histogram_quantile); Grafana RED 面板; Alertmanager (错误率>5%/P99>1s/实例宕机); Jaeger OpenTelemetry 分布式追踪; Node Exporter 机器指标。


## 练习


<!-- Converted from: 9_项目实战 Prometheus  Grafana 监控.html -->
