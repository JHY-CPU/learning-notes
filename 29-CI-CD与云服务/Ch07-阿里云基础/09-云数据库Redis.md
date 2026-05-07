# 云数据库Redis

## 一、概念说明

阿里云云数据库Redis版是兼容开源Redis协议的托管数据库服务，提供高可用、高性能的缓存和数据存储能力。

| 版本 | 架构 | 适用场景 |
|------|------|----------|
| 标准版 | 主从架构 | 中小规模缓存 |
| 集群版 | 分片集群 | 大规模/高吞吐 |
| 读写分离版 | 带只读副本 | 读多写少 |

| 内存类型 | 特点 | 适用 |
|----------|------|------|
| 标准内存 | 高性能 | 热数据 |
| 混合存储 | 内存+SSD | 大容量低成本 |

## 二、具体用法

### 创建实例

```bash
# 创建标准版Redis实例
aliyun r-kvstore CreateInstance \
    --RegionId cn-hangzhou \
    --Capacity 2048 \
    --InstanceClass redis.master.mid.default \
    --EngineVersion 5.0 \
    --VpcId vpc-bp1xxxxxxxx \
    --VSwitchId vsw-bp1xxxxxxxx \
    --Password "RedisPass123!" \
    --InstanceName my-cache \
    --ChargeType PostPaid

# 创建集群版实例
aliyun r-kvstore CreateInstance \
    --RegionId cn-hangzhou \
    --Capacity 32768 \
    --InstanceClass redis.master.large.default \
    --EngineVersion 6.0 \
    --ArchitectureType cluster \
    --ShardCount 4 \
    --VpcId vpc-bp1xxxxxxxx \
    --VSwitchId vsw-bp1xxxxxxxx \
    --Password "RedisPass123!"
```

### 连接和操作

```bash
# 获取连接地址
aliyun r-kvstore DescribeInstances --RegionId cn-hangzhou

# 通过redis-cli连接
redis-cli -h r-bp1xxxxxxxx.redis.rds.aliyuncs.com -p 6379 -a RedisPass123!

# 通过VPN或跳板机连接（推荐方式）
# 先SSH到跳板机，再连接Redis
ssh -L 6379:r-bp1xxxxxxxx.redis.rds.aliyuncs.com:6379 jump-server
redis-cli -h 127.0.0.1 -p 6379
```

### 应用端连接

```python
import redis

# 连接阿里云Redis
r = redis.Redis(
    host='r-bp1xxxxxxxx.redis.rds.aliyuncs.com',
    port=6379,
    password='RedisPass123',
    decode_responses=True,
    socket_timeout=5,
    socket_connect_timeout=5,
    retry_on_timeout=True
)

# 使用连接池
pool = redis.ConnectionPool(
    host='r-bp1xxxxxxxx.redis.rds.aliyuncs.com',
    port=6379,
    password='RedisPass123',
    max_connections=20
)
r = redis.Redis(connection_pool=pool)
```

### 备份和恢复

```bash
# 手动备份（白名单功能）
aliyun r-kvstore CreateBackup --InstanceId r-bp1xxxxxxxx

# 查询备份列表
aliyun r-kvstore DescribeBackups \
    --InstanceId r-bp1xxxxxxxx

# 数据恢复到新实例
aliyun r-kvstore CreateInstance \
    --RegionId cn-hangzhou \
    --BackupId 123456789 \
    --InstanceName restored-redis
```

### 监控和运维

```bash
# 查看实时监控
aliyun r-kvstore DescribeHistoryMonitorValues \
    --InstanceId r-bp1xxxxxxxx \
    --StartTime 2024-01-15T00:00:00Z \
    --EndTime 2024-01-15T01:00:00Z \
    --MonitorKeys "UsedMemory,Connections,HitRate"

# 查看大Key
aliyun r-kvstore DescribeBigKeyRecords \
    --InstanceId r-bp1xxxxxxxx \
    --StartTime 2024-01-15T00:00:00Z \
    --EndTime 2024-01-15T23:59:59Z
```

## 三、注意事项与常见陷阱

1. **白名单配置**：Redis默认拒绝所有连接，必须添加白名单
2. **网络连通**：确保应用和Redis在同一VPC或通过VPN连通
3. **密码安全**：设置强密码，定期更换
4. **内存管理**：关注内存使用率，设置合理的淘汰策略
5. **大Key治理**：定期检查并处理大Key和热Key
6. **连接数限制**：注意最大连接数限制
7. **版本升级**：大版本升级需要充分测试
