# Redis性能概览

## 一、概念说明

Redis是单线程内存数据库，性能通常很高。了解性能瓶颈有助于优化和调优。

## 二、性能指标

```bash
# QPS（每秒查询数）
INFO stats | grep instantaneous_ops_per_sec
# 输出: instantaneous_ops_per_sec:10000

# 延迟
redis-cli --latency
# 输出: min: 0, max: 1, avg: 0.19 (427 samples)

# 内存使用
INFO memory | grep used_memory_human
# 输出: used_memory_human:1.00G
```

## 三、常见瓶颈

```bash
# 1. 内存瓶颈
# 大Key、过多Key、内存碎片

# 2. 网络瓶颈
# 大量小请求、带宽不足

# 3. CPU瓶颈
# 复杂命令、Lua脚本、持久化

# 4. 磁盘IO瓶颈
# AOF fsync、RDB fork
```

## 四、优化方向

```bash
# 1. 减少命令次数 → Pipeline
# 2. 减少数据量 → 合理的数据结构
# 3. 减少网络开销 → 内网部署
# 4. 减少阻塞 → 避免大Key操作
# 5. 使用多线程IO → Redis 6.0+
```

## 五、注意事项

1. **单线程模型**：命令执行是串行的
2. **IO多路复用**：网络IO是并发的
3. **大Key阻塞**：大Key操作阻塞所有命令
4. **持久化影响**：BGSAVE和AOF重写有性能影响
