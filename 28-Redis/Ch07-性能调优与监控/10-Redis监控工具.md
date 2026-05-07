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
4. **多维度监控**：内存、CPU、网络、磁盘
