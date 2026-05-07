# Redis配置文件详解

## 一、概念说明

`redis.conf`是Redis的主要配置文件，包含了Redis服务器运行所需的各种配置参数。合理配置这些参数对Redis的性能、安全性和稳定性至关重要。

## 二、具体用法

### 网络配置

```bash
# 绑定IP地址（默认127.0.0.1，仅本地访问）
bind 127.0.0.1 -::1

# 监听端口（默认6379）
port 6379

# TCP连接队列长度
tcp-backlog 511

# 客户端空闲超时时间（秒，0表示永不超时）
timeout 0

# TCP保活时间
tcp-keepalive 300
```

### 通用配置

```bash
# 是否以守护进程方式运行
daemonize yes

# PID文件路径
pidfile /var/run/redis_6379.pid

# 日志级别（debug/verbose/notice/warning）
loglevel notice

# 日志文件路径
logfile /var/log/redis/redis-server.log

# 数据库数量
databases 16
```

### 快照（RDB）配置

```bash
# 持久化规则：900秒内有1个key变化则保存
save 900 1
# 300秒内有10个key变化则保存
save 300 10
# 60秒内有10000个key变化则保存
save 60 10000

# RDB文件名
dbfilename dump.rdb

# 数据目录
dir /var/lib/redis

# 保存失败时是否停止写入
stop-writes-on-bgsave-error yes

# 是否压缩RDB文件
rdbcompression yes
```

### AOF配置

```bash
# 是否开启AOF持久化
appendonly no

# AOF文件名
appendfilename "appendonly.aof"

# AOF同步策略
# always - 每个写命令都同步（最安全，最慢）
# everysec - 每秒同步一次（推荐）
# no - 由操作系统决定（最快，最不安全）
appendfsync everysec

# AOF重写期间是否同步
no-appendfsync-on-rewrite no

# AOF自动重写触发条件
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb
```

### 内存管理

```bash
# 最大内存限制
maxmemory 1gb

# 内存淘汰策略
# noeviction - 内存满时返回错误（默认）
# allkeys-lru - 从所有key中淘汰最近最少使用
# volatile-lru - 从有过期时间的key中淘汰LRU
# allkeys-random - 随机淘汰
# volatile-random - 从过期key中随机淘汰
# volatile-ttl - 从过期key中淘汰TTL最小的
maxmemory-policy noeviction
```

### 安全配置

```bash
# 设置密码
requirepass yourpassword

# 重命名危险命令
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG "CONFIG_b9f3c5a8"
```

## 三、注意事项与常见陷阱

1. **bind配置**：生产环境不要绑定0.0.0.0，应指定具体IP
2. **maxmemory必须设置**：不设置可能导致系统内存耗尽
3. **save规则**：根据业务需求调整，太频繁影响性能
4. **appendfsync选择**：everysec是安全性与性能的平衡点
5. **危险命令重命名**：FLUSHALL/FLUSHDB在生产环境应禁用或重命名
6. **配置热更新**：部分配置可通过CONFIG SET动态修改，无需重启
