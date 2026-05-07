# 哨兵Sentinel

## 一、概念说明

Redis Sentinel是Redis的高可用解决方案，提供监控、通知和自动故障转移功能。Sentinel集群监控主从节点，在主节点故障时自动将从节点提升为新主节点。

## 二、核心功能

```bash
# 1. 监控（Monitoring）
#    检查主从节点是否正常运行

# 2. 通知（Notification）
#    通过API通知管理员故障事件

# 3. 自动故障转移（Automatic failover）
#    主节点故障时自动提升从节点

# 4. 配置提供者（Configuration provider）
#    客户端通过Sentinel发现主节点地址
```

## 三、具体用法

### 启动Sentinel

```bash
# 方式1：使用redis-sentinel
redis-sentinel /etc/redis/sentinel.conf

# 方式2：使用redis-server
redis-server /etc/redis/sentinel.conf --sentinel

# 最小配置
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 60000
sentinel parallel-syncs mymaster 1
```

### Sentinel配置文件

```bash
# sentinel.conf
port 26379

# 监控主节点
# 格式：sentinel monitor <master-name> <ip> <port> <quorum>
sentinel monitor mymaster 192.168.1.100 6379 2

# 主节点密码
sentinel auth-pass mymaster password

# 主观下线时间（毫秒）
sentinel down-after-milliseconds mymaster 5000

# 故障转移超时时间（毫秒）
sentinel failover-timeout mymaster 60000

# 故障转移时并行同步的从节点数
sentinel parallel-syncs mymaster 1
```

### Sentinel管理命令

```bash
# 连接Sentinel
redis-cli -p 26379

# 查看主节点信息
SENTINEL masters
SENTINEL master mymaster

# 查看从节点
SENTINEL replicas mymaster

# 查看Sentinel节点
SENTINEL sentinels mymaster

# 获取当前主节点地址
SENTINEL get-master-addr-by-name mymaster
# 输出: 1) "192.168.1.100" 2) "6379"

# 手动触发故障转移
SENTINEL failover mymaster
```

## 四、故障转移过程

```
Sentinel检测到主节点主观下线
    │
    ▼
多Sentinel确认客观下线（达到quorum）
    │
    ▼
选举领头Sentinel
    │
    ▼
领头Sentinel选择新主节点
    │
    ▼
执行故障转移
    ├── 将从节点提升为主节点
    ├── 修改其他从节点的主节点
    └── 通知客户端新主节点地址
```

## 五、注意事项与常见陷阱

1. **Sentinel数量**：建议至少3个Sentinel，奇数个
2. **quorum设置**：建议设置为Sentinel数量的一半+1
3. **脑裂问题**：网络分区可能导致多个主节点
4. **客户端支持**：需要支持Sentinel的客户端库
5. **配置持久化**：Sentinel会自动更新配置文件
6. **故障转移时间**：通常需要10-30秒
