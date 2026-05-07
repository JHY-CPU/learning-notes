# INFO命令详解

## 一、概念说明

INFO命令返回Redis服务器的各种信息和统计数据，是监控和调优的重要工具。

## 二、主要Section

```bash
# 服务器信息
INFO server
# redis_version, redis_mode, process_id, tcp_port

# 客户端信息
INFO clients
# connected_clients, blocked_clients, maxclients

# 内存信息
INFO memory
# used_memory, maxmemory, mem_fragmentation_ratio

# 持久化信息
INFO persistence
# rdb_last_bgsave_status, aof_last_rewrite_status

# 统计信息
INFO stats
# total_connections_received, instantaneous_ops_per_sec
# keyspace_hits, keyspace_misses

# 复制信息
INFO replication
# role, connected_slaves, master_link_status

# CPU信息
INFO cpu
# used_cpu_sys, used_cpu_user

# 键空间
INFO keyspace
# db0:keys=1000,expires=500
```

## 三、关键指标

```bash
# 命中率
keyspace_hits / (keyspace_hits + keyspace_misses)

# 内存使用率
used_memory / maxmemory

# 碎片率
mem_fragmentation_ratio

# 每秒操作数
instantaneous_ops_per_sec

# 连接数
connected_clients / maxclients
```

## 四、监控脚本

```python
import redis

r = redis.Redis()

info = r.info()

# 关键指标
print(f"命中率: {info['keyspace_hits']/(info['keyspace_hits']+info['keyspace_misses'])*100:.1f}%")
print(f"内存: {info['used_memory_human']}")
print(f"碎片率: {info['mem_fragmentation_ratio']:.2f}")
print(f"QPS: {info['instantaneous_ops_per_sec']}")
print(f"连接数: {info['connected_clients']}")
```

## 五、注意事项

1. **定期检查**：每天检查关键指标
2. **趋势分析**：记录历史数据，分析趋势
3. **告警设置**：关键指标超过阈值告警
4. **Section选择**：只获取需要的section减少开销
