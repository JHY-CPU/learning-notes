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

## 六、性能测试方法

```bash
# 延迟测试
redis-cli --latency
# 输出: min: 0, max: 3, avg: 0.19 (427 samples)

redis-cli --latency-history -i 1
# 每秒输出一次延迟统计

redis-cli --latency-dist
# 延迟分布直方图

# QPS测试
redis-benchmark -t set,get -n 100000 -c 50

# 内存使用分析
MEMORY USAGE key_name
MEMORY STATS
MEMORY DOCTOR

# 命令统计
INFO commandstats
# cmdstat_get:calls=10000,usec=5000,usec_per_call=0.50
# cmdstat_set:calls=5000,usec=3000,usec_per_call=0.60
```

## 七、性能调优清单

```bash
# 1. 配置优化
- [ ] maxmemory已设置
- [ ] maxmemory-policy已配置
- [ ] maxclients已调整
- [ ] timeout已设置
- [ ] tcp-backlog已优化

# 2. 数据结构优化
- [ ] 使用Hash存储对象（ziplist编码）
- [ ] Key命名简洁
- [ ] Value使用紧凑格式
- [ ] 避免大Key（>10MB）
- [ ] 避免热Key（分片处理）

# 3. 操作优化
- [ ] 使用Pipeline批量操作
- [ ] 避免KEYS *，使用SCAN
- [ ] 大Key删除使用UNLINK
- [ ] 复杂操作使用Lua脚本

# 4. 架构优化
- [ ] 使用连接池
- [ ] 部署在内网
- [ ] 配置主从复制
- [ ] 考虑集群分片

# 5. 监控优化
- [ ] 部署监控系统
- [ ] 配置慢查询日志
- [ ] 设置告警规则
- [ ] 定期分析性能指标
```

## 八、Redis 6.0+多线程IO

```bash
# 配置多线程IO
io-threads 4              # IO线程数（建议CPU核数-1）
io-threads-do-reads yes   # 读操作也使用多线程

# 注意事项
# 1. 命令执行仍是单线程
# 2. 多线程只处理网络IO
# 3. CPU密集场景效果明显
# 4. 需要Redis 6.0+

# 性能提升
# 单线程：100K QPS
# 4线程IO：约 300-400K QPS
# 提升约3-4倍
```
