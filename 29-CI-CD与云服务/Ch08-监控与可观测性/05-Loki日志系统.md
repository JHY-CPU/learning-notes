# Loki日志系统

## 一、概念说明

Loki是Grafana Labs开发的日志聚合系统，设计为"像Prometheus但用于日志"。相比ELK更轻量，只索引日志的元数据而非全文。

| 概念 | 说明 |
|------|------|
| Log Stream | 同一标签集的日志流 |
| Label | 索引的元数据键值对 |
| Chunk | 日志数据存储单元 |
| Tenant | 多租户隔离 |

## 二、具体用法

### 部署Loki

```yaml
# docker-compose.yml
version: '3.8'
services:
  loki:
    image: grafana/loki:2.9.0
    command: -config.file=/etc/loki/local-config.yaml
    ports:
      - 3100:3100
    volumes:
      - loki-data:/loki

  promtail:
    image: grafana/promtail:2.9.0
    command: -config.file=/etc/promtail/config.yml
    volumes:
      - /var/log:/var/log:ro
      - ./promtail-config.yml:/etc/promtail/config.yml

volumes:
  loki-data:
```

### Promtail配置

```yaml
# promtail-config.yml
server:
  http_listen_port: 9080

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # 应用日志
  - job_name: app-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: web-app
          env: production
          __path__: /var/log/app/*.log

    pipeline_stages:
      - json:
          expressions:
            level: level
            message: msg
            timestamp: ts
      - labels:
          level:
      - timestamp:
          source: timestamp
          format: RFC3339

  # Nginx日志
  - job_name: nginx-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: nginx
          __path__: /var/log/nginx/*.log
    pipeline_stages:
      - regex:
          expression: '.*(?P<status>\d{3}) (?P<duration>\d+)ms'
      - labels:
          status:
```

### LogQL查询

```bash
# 基本查询
{job="web-app"}

# 标签过滤
{job="web-app", env="production"}
{job="web-app", level="error"}

# 内容过滤
{job="web-app"} |= "timeout"
{job="web-app"} |~ "error|exception"
{job="web-app"} != "healthcheck"

# JSON解析
{job="web-app"} | json | level="ERROR"
{job="web-app"} | json | duration > 500

# 统计查询
count_over_time({job="web-app"} |= "error" [5m])

# 错误率
sum(rate({job="web-app"} | json | level="ERROR" [5m])) / sum(rate({job="web-app"} [5m]))

# 按标签聚合
sum(count_over_time({job="web-app"} | json | level="ERROR" [1h])) by (service)

# 格式化输出
{job="web-app"} | json | line_format "{{.timestamp}} {{.level}} {{.message}}"
```

### Grafana集成

```yaml
# Grafana数据源配置
datasources:
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    jsonData:
      maxLines: 1000
      derivedFields:
        - datasourceUid: jaeger
          matcherRegex: 'trace_id=(\w+)'
          name: TraceID
          url: '$${__value.raw}'
```

### Python直接推送到Loki

```python
import logging
from logging_loki import LokiHandler

# 配置Loki Handler
handler = LokiHandler(
    url="http://loki:3100/loki/api/v1/push",
    tags={"application": "my-app", "environment": "production"},
    version="1",
)

logger = logging.getLogger("my-app")
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# 发送日志
logger.info("用户操作完成", extra={"tags": {"user_id": "12345", "action": "checkout"}})
logger.error("支付失败", extra={"tags": {"order_id": "ORDER-001", "error_code": "E001"}})
```

## 三、注意事项与常见陷阱

1. **标签选择**：标签值不能太多（避免高基数），只用于有限的维度
2. **日志行大小**：单行日志最大256KB，超大日志考虑截断
3. **保留策略**：配置retention_period自动清理旧日志
4. **存储性能**：使用SSD存储提升查询性能
5. **多租户**：生产环境启用多租户隔离
6. **Promtail性能**：日志量大时调整batch大小和发送频率
7. **告警集成**：通过Grafana告警规则基于LogQL触发告警
