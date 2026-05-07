# AOF追加日志详解

## 一、概念说明

AOF（Append Only File）将每个写操作以Redis协议格式追加到日志文件中。重启时重放所有命令来恢复数据。AOF提供比RDB更好的数据安全性。

## 二、具体用法

### 开启AOF

```bash
# redis.conf 配置
appendonly yes

# AOF文件名
appendfilename "appendonly.aof"

# AOF文件目录
dir /var/lib/redis
```

### 同步策略

```bash
# appendfsync 配置
# always - 每个写命令都fsync（最安全，最慢）
appendfsync always

# everysec - 每秒fsync一次（推荐）
appendfsync everysec

# no - 由操作系统决定fsync时机（最快，最不安全）
appendfsync no

# 性能与安全性权衡
# always: 零数据丢失，但性能最差
# everysec: 最多丢失1秒数据，性能良好
# no: 可能丢失更多数据，性能最好
```

### AOF文件格式

```bash
# AOF文件内容示例
*2
$6
SELECT
$1
0
*3
$3
SET
$4
name
$5
hello
*3
$3
SET
$4
key2
$5
world
```

```bash
# 查看AOF文件内容
cat /var/lib/redis/appendonly.aof

# 验证AOF文件
redis-check-aof /var/lib/redis/appendonly.aof
# 输出: AOF文件验证结果
```

### AOF加载

```bash
# 重启时自动加载AOF文件
redis-server /etc/redis/redis.conf

# 查看加载状态
INFO persistence | grep aof_last_bgrewrite_status
# 输出: aof_last_bgrewrite_status:ok

# 如果AOF和RDB同时存在
# Redis优先加载AOF文件（数据更完整）
```

## 三、AOF工作原理

```
客户端写命令
    │
    ▼
追加到AOF缓冲区
    │
    ▼
根据appendfsync策略写入磁盘
    │
    ▼
AOF文件增长
    │
    ▼
触发AOF重写（压缩）
```

## 四、配置优化

```bash
# AOF重写期间是否禁止fsync
no-appendfsync-on-rewrite no
# 设置为yes可以减少重写期间的延迟
# 但可能丢失更多数据

# AOF文件增长百分比触发重写
auto-aof-rewrite-percentage 100
# 文件大小翻倍时触发重写

# AOF文件最小大小触发重写
auto-aof-rewrite-min-size 64mb
# 文件小于64MB时不重写
```

## 五、注意事项与常见陷阱

1. **AOF文件可能很大**：需要定期重写
2. **重放开销**：大量命令重放恢复慢
3. **fsync策略选择**：everysec是最佳平衡
4. **磁盘IO压力**：always策略对磁盘要求高
5. **命令幂等性**：确保写命令幂等（避免INCR等不幂等命令的数据问题）
6. **AOF文件损坏**：使用redis-check-aof修复
