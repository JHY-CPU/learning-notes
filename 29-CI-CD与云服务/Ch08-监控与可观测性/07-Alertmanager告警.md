# Alertmanager告警

## 一、概念说明

Alertmanager是Prometheus生态中的告警管理组件，负责告警的去重、分组、静默和路由通知。

| 概念 | 说明 |
|------|------|
| Alert | 触发的告警实例 |
| Route | 告警路由规则 |
| Receiver | 通知接收者 |
| Silence | 临时静默规则 |
| Inhibition | 告警抑制规则 |
| Group | 告警分组 |

## 二、具体用法

### Alertmanager配置

```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m
  smtp_smarthost: 'smtp.example.com:587'
  smtp_from: 'alertmanager@example.com'
  smtp_auth_username: 'alertmanager'
  smtp_auth_password: 'password'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 1m
  repeat_interval: 4h
  receiver: 'default-webhook'

  routes:
    # 紧急告警
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 5s
      repeat_interval: 1h

    # 警告级别
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 12h

receivers:
  - name: 'default-webhook'
    webhook_configs:
      - url: 'http://alert-webhook:5001/webhook'

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@example.com'
        send_resolved: true
    webhook_configs:
      - url: 'https://oapi.dingtalk.com/robot/send?access_token=xxx'
    pagerduty_configs:
      - service_key: 'your-pd-key'

  - name: 'warning-alerts'
    email_configs:
      - to: 'team@example.com'
        send_resolved: true

# 告警抑制
inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster']
```

### 钉钉告警配置

```yaml
# Webhook方式
receivers:
  - name: 'dingtalk'
    webhook_configs:
      - url: 'http://dingtalk-webhook:8060/dingtalk/ops/send'
        send_resolved: true
```

```python
# dingtalk-webhook服务（Python实现）
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

DINGTALK_WEBHOOK = "https://oapi.dingtalk.com/robot/send?access_token=xxx"

@app.route('/dingtalk/<group>/send', methods=['POST'])
def send_alert(group):
    data = request.json
    alerts = data.get('alerts', [])

    for alert in alerts:
        status = "触发" if alert['status'] == 'firing' else "恢复"
        title = alert['labels'].get('alertname', 'Unknown')
        description = alert['annotations'].get('description', 'No description')

        markdown = f"## {status}: {title}\n\n{description}\n\n"

        requests.post(DINGTALK_WEBHOOK, json={
            "msgtype": "markdown",
            "markdown": {
                "title": f"{status}: {title}",
                "text": markdown
            }
        })

    return jsonify({"status": "ok"})
```

### 告警规则示例

```yaml
# alert_rules.yml
groups:
  - name: infrastructure
    rules:
      - alert: NodeDown
        expr: up{job="node"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "节点 {{ $labels.instance }} 不可达"
          description: "节点已离线超过2分钟"

      - alert: HighMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "节点 {{ $labels.instance }} 内存使用率过高"
          description: "当前内存使用率: {{ $value | printf \"%.1f\" }}%"

      - alert: DiskSpaceLow
        expr: (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "节点 {{ $labels.instance }} 磁盘空间不足"
          description: "磁盘使用率: {{ $value | printf \"%.1f\" }}%"

  - name: application
    rules:
      - alert: HighErrorRate
        expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) / sum(rate(http_requests_total[5m])) by (service) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "{{ $labels.service }} 错误率过高"
          description: "错误率: {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "{{ $labels.service }} 延迟过高"
          description: "P95延迟: {{ $value }}s"
```

### 静默管理

```bash
# 创建静默规则
curl -X POST http://alertmanager:9093/api/v1/silences -d '{
    "matchers": [
        {"name": "alertname", "value": "HighCPU", "isRegex": false},
        {"name": "instance", "value": "test-.*", "isRegex": true}
    ],
    "startsAt": "2024-01-15T10:00:00Z",
    "endsAt": "2024-01-15T18:00:00Z",
    "createdBy": "admin",
    "comment": "维护窗口"
}'

# 查看活跃告警
curl http://alertmanager:9093/api/v2/alerts

# 查看静默规则
curl http://alertmanager:9093/api/v2/silences
```

## 三、注意事项与常见陷阱

1. **告警分级**：清晰定义critical/warning/info级别
2. **分组策略**：合理的group_by避免告警风暴
3. **静默窗口**：计划维护时提前设置静默规则
4. **抑制规则**：高级别告警抑制低级别告警
5. **告警恢复**：配置send_resolved自动发送恢复通知
6. **通知渠道**：多渠道备份避免通知丢失
7. **告警模板**：使用Go模板自定义告警消息格式
