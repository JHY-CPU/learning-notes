# RDB快照详解

## 一、概念说明

RDB（Redis Database）是Redis默认的持久化方式，通过定期将内存数据快照保存到磁盘文件来实现持久化。RDB文件是紧凑的二进制格式，适合备份和恢复。

## 二、具体用法

### 手动生成快照

```bash
# SAVE - 同步保存（阻塞主线程）
SAVE
# 输出: OK
# 阻塞直到RDB文件生成完成
# 生产环境禁止使用

# BGSAVE - 异步保存（推荐）
BGSAVE
# 输出: Background saving started
# 后台fork子进程保存，不阻塞主线程

# 查看保存状态
LASTSAVE
# 输出: (integer) 1704067200（上次保存时间戳）

# 查看RDB信息
INFO persistence
# 输出: rdb_last_bgsave_status:ok
#       rdb_last_bgsave_time_sec:1
```

### 自动保存配置

```bash
# redis.conf 配置自动保存规则
# 格式：save <seconds> <changes>
save 900 1       # 900秒（15分钟）内有1个key变化
save 300 10      # 300秒（5分钟）内有10个key变化
save 60 10000    # 60秒内有10000个key变化

# 禁用RDB
save ""

# RDB文件名
dbfilename dump.rdb

# RDB文件目录
dir /var/lib/redis
```

### RDB压缩与校验

```bash
# 启用压缩（默认开启）
rdbcompression yes

# 启用CRC64校验（默认开启）
rdbchecksum yes

# 使用LZF压缩
# 减少磁盘空间占用，但增加CPU开销
```

## 三、RDB工作原理

```
触发BGSAVE
    │
    ▼
fork()创建子进程 ──────── 主进程继续处理请求
    │
    ▼
子进程遍历内存数据
    │
    ▼
写入临时RDB文件
    │
    ▼
替换旧的RDB文件
    │
    ▼
子进程退出，通知主进程
```

## 四、查看与恢复

```bash
# 查看RDB文件信息
redis-check-rdb /var/lib/redis/dump.rdb
# 输出: RDB文件的详细信息

# 恢复RDB文件
# 将RDB文件放到Redis数据目录
# 启动Redis即可自动恢复

# 使用RDB文件迁移数据
# 1. 源服务器BGSAVE
# 2. 复制dump.rdb到目标服务器
# 3. 目标服务器启动恢复
```

## 五、注意事项与常见陷阱

1. **BGSAVE期间写操作**：使用COW（Copy-On-Write），子进程看到的是fork时的数据快照
2. **fork开销**：内存越大，fork越慢，可能造成短暂阻塞
3. **磁盘空间**：确保磁盘空间足够存放RDB文件
4. **数据丢失风险**：两次快照间的数据会丢失
5. **禁用SAVE命令**：生产环境应重命名或禁用SAVE命令
6. **备份策略**：定期将RDB文件备份到远程存储
