# Grafana可视化

## 一、概念说明

Grafana是开源的数据可视化和监控平台，支持多种数据源。用于创建仪表板和告警。

| 数据源 | 用途 |
|--------|------|
| Prometheus | 时序指标 |
| Elasticsearch | 日志数据 |
| Loki | 日志数据 |
| MySQL/PostgreSQL | SQL查询 |
| Jaeger | 链路追踪 |
| CloudWatch | AWS监控 |

## 二、具体用法

### 安装和配置

```bash
# Docker方式运行
docker run -d \
    --name grafana \
    -p 3000:3000 \
    -v grafana-data:/var/lib/grafana \
    -e GF_SECURITY_ADMIN_PASSWORD=admin123 \
    grafana/grafana:latest

# 通过Provisioning自动配置数据源
```

```yaml
# provisioning/datasources/prometheus.yml
apiVersion: 1
datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true

  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
```

### 仪表板配置（JSON Model）

```json
{
    "title": "应用监控仪表板",
    "panels": [
        {
            "title": "QPS",
            "type": "timeseries",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "sum(rate(http_requests_total[5m])) by (endpoint)",
                "legendFormat": "{{ endpoint }}"
            }],
            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
        },
        {
            "title": "P95延迟",
            "type": "timeseries",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
                "legendFormat": "P95"
            }],
            "fieldConfig": {
                "defaults": {
                    "unit": "s"
                }
            }
        },
        {
            "title": "错误率",
            "type": "gauge",
            "datasource": "Prometheus",
            "targets": [{
                "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100"
            }],
            "fieldConfig": {
                "defaults": {
                    "unit": "percent",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 1},
                            {"color": "red", "value": 5}
                        ]
                    }
                }
            }
        }
    ],
    "refresh": "10s",
    "time": {"from": "now-1h", "to": "now"}
}
```

### 常用PromQL面板

```bash
# CPU使用率
100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)

# 内存使用率
(1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100

# 磁盘使用率
(1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100

# 网络流量
rate(node_network_receive_bytes_total[5m])

# 容器CPU使用
sum(rate(container_cpu_usage_seconds_total{name=~".+"}[5m])) by (name)

# 数据库连接数
pg_stat_activity_count

# Redis命中率
redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total)
```

### Grafana告警

```json
{
    "name": "高错误率告警",
    "condition": "C",
    "data": [
        {
            "refId": "A",
            "relativeTimeRange": {"from": 300, "to": 0},
            "model": {
                "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) / sum(rate(http_requests_total[5m])) * 100"
            }
        },
        {
            "refId": "C",
            "model": {
                "type": "threshold",
                "conditions": [{
                    "evaluator": {"type": "gt", "params": [5]},
                    "operator": {"type": "and"}
                }]
            }
        }
    ]
}
```

### 代码管理仪表板

```yaml
# provisioning/dashboards/dashboard.yml
apiVersion: 1
providers:
  - name: 'Default'
    folder: 'Application'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: true
```

## 三、注意事项与常见陷阱

1. **数据源配置**：使用Provisioning管理数据源，便于版本控制
2. **仪表板模板化**：使用变量和模板实现动态仪表板
3. **性能优化**：避免单个仪表板太多面板，控制查询频率
4. **权限管理**：配置团队和角色权限
5. **仪表板版本**：仪表板JSON纳入Git版本控制
6. **告警通知**：配置通知渠道（邮件、钉钉、Slack）
7. **插件管理**：仅安装需要的插件，注意安全风险
