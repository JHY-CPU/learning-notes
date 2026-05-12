# Redis监控工具

## 一、概念说明

使用专业的监控工具可以实时监控Redis的状态，设置告警规则，及时发现问题。

## 二、Prometheus + Grafana

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

# redis_exporter配置
# docker run -d --name redis_exporter \
#   -p 9121:9121 \
#   oliver006/redis_exporter \
#   --redis.addr=redis://192.168.1.100:6379
```

## 三、Grafana仪表板

```bash
# 推荐仪表板
# 1. Redis Dashboard (ID: 763)
# 2. Redis Exporter Quickstart (ID: 11835)

# 关键面板
# - 内存使用趋势
# - 命中率趋势
# - 连接数趋势
# - 命令执行QPS
# - 慢查询数量
# - 主从复制延迟
```

## 四、告警规则

```yaml
# 命中率告警
- alert: RedisHitRateLow
  expr: redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) < 0.8
  for: 5m

# 内存告警
- alert: RedisMemoryHigh
  expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
  for: 5m

# 连接数告警
- alert: RedisConnectionsHigh
  expr: redis_connected_clients > 1000
  for: 5m
```

## 五、其他工具

```bash
# redis-stat
redis-stat --server 192.168.1.100:6379

# RedisLive
# 基于Python的Web监控工具

# DataDog
# 商业APM工具，支持Redis监控

# Datadog
# 全栈APM监控
```

## 六、注意事项

1. **监控粒度**：合适的采集间隔（15s-60s）
2. **数据保留**：历史数据保留策略
3. **告警收敛**：避免告警风暴
## 七、Grafana仪表板配置

```json
{
  "dashboard": {
    "title": "Redis Monitoring",
    "panels": [
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_memory_used_bytes",
            "legendFormat": "Used Memory"
          },
          {
            "expr": "redis_memory_max_bytes",
            "legendFormat": "Max Memory"
          }
        ]
      },
      {
        "title": "Hit Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100"
          }
        ]
      },
      {
        "title": "Commands/sec",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(redis_commands_processed_total[1m])",
            "legendFormat": "QPS"
          }
        ]
      },
      {
        "title": "Connected Clients",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_connected_clients"
          }
        ]
      }
    ]
  }
}
```

## 八、告警规则详解

```yaml
# Prometheus告警规则
groups:
  - name: redis_alerts
    rules:
      # 命中率告警
      - alert: RedisHitRateLow
        expr: redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis命中率低于80%"
          description: "当前命中率: {{ $value | humanizePercentage }}"
      
      # 内存告警
      - alert: RedisMemoryHigh
        expr: redis_memory_used_bytes / redis_memory_max_bytes > 0.8
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Redis内存使用超过80%"
          description: "当前使用率: {{ $value | humanizePercentage }}"
      
      # 连接数告警
      - alert: RedisConnectionsHigh
        expr: redis_connected_clients > 5000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis连接数超过5000"
      
      # 慢查询告警
      - alert: RedisSlowQueries
        expr: increase(redis_slowlog_length[5m]) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis慢查询突增"
      
      # 主从延迟告警
      - alert: RedisReplicationLag
        expr: redis_replication_offset_diff > 10000000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Redis主从延迟过大"
      
      # 拒绝连接告警
      - alert: RedisRejectedConnections
        expr: increase(redis_rejected_connections_total[5m]) > 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis拒绝连接"
```

## 九、监控数据保留策略

```bash
# Prometheus数据保留
# 默认保留15天
--storage.tsdb.retention.time=15d

# 按时间精度保留
# 最近2小时: 5秒精度
# 最近24小时: 1分钟精度
# 最近15天: 5分钟精度

# 数据清理
# Prometheus自动清理过期数据
# 可配置保留大小
--storage.tsdb.retention.size=50GB
```
