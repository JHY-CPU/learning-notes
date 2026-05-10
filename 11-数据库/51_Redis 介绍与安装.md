# Redis 介绍与安装


## ⚡ Redis 介绍与安装


Redis 概述 (内存数据库/缓存/消息中间件)、数据结构概览、安装 (Docker/源码/macOS)、redis-cli 基本使用、配置与安全。


## Redis 是什么?


```
// ========== Redis ==========
// Remote Dictionary Server (远程字典服务)

// ========== 核心特点 ==========
// 1. 内存数据库: 读写速度极快 (10万+ QPS)
// 2. 丰富数据结构: String/List/Set/Hash/ZSet
// 3. 持久化: RDB 快照 + AOF 日志
// 4. 高可用: 主从复制 + Sentinel 哨兵 + Cluster
// 5. 原子操作: 单线程模型 (6.0+ 多线程网络IO)

// ========== 典型应用场景 ==========
// ┌──────────────┬─────────────────────────┐
// │ 场景         │ 使用方式                │
// ├──────────────┼─────────────────────────┤
// │ 缓存         │ String 缓存热点数据     │
// │ 会话         │ String 存 session       │
// │ 计数器       │ INCR 计数               │
// │ 限流         │ INCR + EXPIRE 窗口      │
// │ 排行榜       │ ZSet 有序集合           │
// │ 消息队列     │ List (LPUSH/BRPOP)      │
// │ 布隆过滤器   │ 插件: RedisBloom        │
// │ 分布式锁     │ SETNX + Lua             │
// │ 地理位置     │ GeoHash (GEOADD)        │
// └──────────────┴─────────────────────────┘

// ========== 与其他数据库对比 ==========
// ┌────────┬──────────┬───────────┬──────────┐
// │        │ Redis    │ Memcached │  MySQL   │
// ├────────┼──────────┼───────────┼──────────┤
// │ 类型   │ 内存+持久 │ 内存      │ 磁盘    │
// │ 数据结构│ 丰富     │ String    │ 关系型  │
// │ 持久化 │ RDB/AOF  │ ❌        │ ✅      │
// │ 集群   │ 原生     │ 客户端   │ 中间件  │
// │ QPS    │ 10万+    │ 10万+    │ 数千    │
// └────────┴──────────┴───────────┴──────────┘
```


## 安装 Redis


```
// ========== 安装方式 ==========

// ========== 1. Docker (推荐) ==========
docker run -d --name redis -p 6379:6379 redis:7-alpine

// 带密码和持久化:
docker run -d --name redis \
    -p 6379:6379 \
    -v redis-data:/data \
    redis:7-alpine \
    redis-server --requirepass mypassword --appendonly yes

// ========== 2. macOS (Homebrew) ==========
brew install redis
brew services start redis
brew services stop redis

// ========== 3. Ubuntu / Debian ==========
sudo apt update
sudo apt install redis-server
sudo systemctl enable redis
sudo systemctl start redis

// ========== 4. 源码编译 ==========
wget https://download.redis.io/redis-stable.tar.gz
tar xzf redis-stable.tar.gz
cd redis-stable
make
make install

// 启动:
redis-server

// ========== 5. Windows ==========
// 官方不支持 Windows, 使用 WSL2
// 或 Microsoft Archive 版本
// 推荐 Docker 方式
```


## redis-cli 基本使用


```
// ========== redis-cli 命令 ==========

// 连接
redis-cli                              // 默认 localhost:6379
redis-cli -h 192.168.1.100 -p 6380    // 指定主机/端口
redis-cli -a mypassword                // 带密码

// ========== Ping 测试 ==========
ping        // 返回 PONG

// ========== Key 操作 ==========
SET name "Alice"        // 设置键值
GET name                // 获取: "Alice"
EXISTS name             // 是否存在: 1
TYPE name               // 类型: string
DEL name                // 删除: 1
KEYS *                  // 所有键 (生产禁用!)
SCAN 0 MATCH user:*     // 游标扫描 (生产用)

// ========== 信息查看 ==========
INFO                     // 服务器信息
INFO memory              // 内存信息
INFO stats               // 统计信息
DBSIZE                   // 键数量
CLIENT LIST              // 客户端列表
SLOWLOG GET 10           // 慢查询

// ========== 数据库切换 ==========
// Redis 默认 16 个数据库 (0-15)
SELECT 0                 // 选择 db 0
SELECT 1                 // 切换 db 1
FLUSHDB                  // 清空当前库
FLUSHALL                 // 清空所有库 (慎用!)

// ========== 退出 ==========
exit
quit
```


## 配置 Redis


```
// ========== redis.conf 核心配置 ==========

// redis.conf 文件位置:
// /etc/redis/redis.conf (Linux)
// /usr/local/etc/redis.conf (macOS Homebrew)

// ========== 基本配置 ==========
# 绑定地址 (默认仅本地)
bind 127.0.0.1

# 端口
port 6379

# 守护进程
daemonize yes

# PID 文件
pidfile /var/run/redis_6379.pid

# 日志级别: debug/verbose/notice/warning
loglevel notice

# 日志文件
logfile /var/log/redis/redis.log

# 数据库数量
databases 16

// ========== 安全配置 ==========
# 密码
requirepass your_strong_password

# 重命名危险命令 (防注入)
rename-command FLUSHALL ""
rename-command FLUSHDB ""
rename-command CONFIG ""

// ========== 内存配置 ==========
# 最大内存
maxmemory 1gb

# 淘汰策略 (见内存淘汰文件)
maxmemory-policy allkeys-lru

// ========== 持久化配置 ==========
# RDB
save 900 1        # 900 秒内 1 次修改
save 300 10       # 300 秒内 10 次修改
save 60 10000     # 60 秒内 10000 次修改

# AOF
appendonly yes
appendfsync everysec

// ========== 运行时修改配置 ==========
CONFIG GET maxmemory          # 查看配置
CONFIG SET maxmemory 2gb      # 修改配置 (即时生效)
CONFIG REWRITE                # 写入配置文件
```


> **Note:** 💡 Redis 要点: 内存数据库 10万+ QPS; 5 种核心数据结构; Docker 安装最方便; KEYS 命令生产禁用用 SCAN; 数据库 0-15; 生产必须设密码和 rename-command; maxmemory 限制内存使用。


## 练习


<!-- Converted from: 51_Redis 介绍与安装.html -->
