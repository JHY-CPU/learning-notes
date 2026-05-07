# Redis集群架构概览

## 一、概念说明

Redis提供三种主要的集群架构：主从复制、哨兵模式和Cluster集群模式。每种架构适用于不同的场景和需求。

## 二、架构模式详解

### 主从复制（Master-Slave）

```bash
# 架构特点
# - 一个主节点，多个从节点
# - 主节点负责写操作，从节点负责读操作
# - 数据从主节点复制到从节点
# - 不支持自动故障转移

# 配置从节点
# 在从节点配置文件中添加
replicaof 192.168.1.100 6379

# 或者命令行配置
REPLICAOF 192.168.1.100 6379

# 查看复制状态
INFO replication
# 输出: role:slave
#       master_host:192.168.1.100
#       master_link_status:up
```

### 哨兵模式（Sentinel）

```bash
# 架构特点
# - 在主从复制基础上增加哨兵
# - 哨兵监控主从节点状态
# - 支持自动故障转移
# - 客户端通过哨兵发现主节点

# sentinel.conf配置
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 60000
sentinel parallel-syncs mymaster 1

# 启动哨兵
redis-sentinel /etc/redis/sentinel.conf

# 查看哨兵状态
redis-cli -p 26379 SENTINEL masters
```

### Cluster集群模式

```bash
# 架构特点
# - 数据分片到多个节点
# - 使用16384个哈希槽
# - 支持自动故障转移
# - 支持水平扩展

# 槽位分配
# CRC16(key) % 16384 = slot

# 查看集群状态
redis-cli -c CLUSTER INFO
# 输出: cluster_state:ok
#       cluster_slots_assigned:16384

redis-cli -c CLUSTER NODES
# 输出: 节点列表和槽位分配

# 数据分片示例
redis-cli -c SET key1 "value1"
# 输出: -> Redirected to slot [9189] located at 192.168.1.103:6379
#       OK
```

## 三、架构对比

| 特性 | 主从复制 | 哨兵模式 | Cluster |
|------|----------|----------|---------|
| 数据分片 | 不支持 | 不支持 | 支持 |
| 自动故障转移 | 不支持 | 支持 | 支持 |
| 水平扩展 | 不支持 | 不支持 | 支持 |
| 配置复杂度 | 低 | 中 | 高 |
| 适用场景 | 小型系统 | 中型系统 | 大型系统 |

## 四、注意事项与常见陷阱

1. **主从复制延迟**：异步复制可能有数据延迟
2. **哨兵脑裂**：网络分区可能导致多个主节点
3. **Cluster槽位迁移**：数据迁移期间可能有短暂不可用
4. **客户端支持**：需要支持集群的客户端库
5. **多Key操作**：Cluster模式下多Key操作需要在同一个槽
6. **最小节点数**：Cluster至少需要3个主节点
