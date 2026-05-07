# AOF重写

## 一、概念说明

AOF重写是压缩AOF文件的过程。随着时间推移，AOF文件会包含大量冗余命令（如多次SET同一个key）。重写通过读取当前数据状态，生成最小化的命令序列来压缩文件。

## 二、具体用法

### 手动触发重写

```bash
# 手动触发AOF重写
BGREWRITEAOF
# 输出: Background append only file rewriting started

# 查看重写状态
INFO persistence | grep aof_last_bgrewrite_status
# 输出: aof_last_bgrewrite_status:ok

# 查看重写进度
INFO persistence | grep aof_rewrite_in_progress
# 输出: aof_rewrite_in_progress:1（正在重写）
```

### 自动重写配置

```bash
# redis.conf 配置自动重写

# 文件增长百分比触发重写
auto-aof-rewrite-percentage 100
# 当前AOF文件大小比上次重写后大100%时触发

# 最小文件大小
auto-aof-rewrite-min-size 64mb
# AOF文件小于64MB时不触发重写

# 示例
# 上次重写后AOF大小：50MB
# 当前AOF大小：100MB
# 增长百分比：(100-50)/50 = 100%
# 触发条件满足，开始重写
```

### 重写工作原理

```bash
# 重写过程
# 1. fork子进程
# 2. 子进程读取当前内存数据
# 3. 生成新的AOF文件（最小命令序列）
# 4. 重写期间的新写命令追加到AOF重写缓冲区
# 5. 子进程完成后，将重写缓冲区内容追加到新AOF文件
# 6. 原子性替换旧AOF文件

# 示例：10次SET同一个key
SET mykey "value1"
SET mykey "value2"
SET mykey "value3"
# 重写后只保留最后一条
SET mykey "value3"
```

## 三、重写优化

```bash
# 重写期间避免fsync阻塞
no-appendfsync-on-rewrite yes
# 重写期间禁止fsync，减少延迟
# 但可能丢失数据

# 监控重写耗时
INFO persistence | grep aof_last_rewrite_time_sec
# 输出: aof_last_rewrite_time_sec:5

# 监控COW内存使用
INFO persistence | grep aof_last_cow_size
# 输出: aof_last_cow_size:1048576
```

## 四、重写场景分析

```bash
# 场景1：频繁更新同一key
# 重写前：100条SET命令
# 重写后：1条SET命令

# 场景2：大量删除操作
# 重写前：1000条写命令 + 500条DEL命令
# 重写后：500条SET命令

# 场景3：批量导入后
# 重写前：100万条写命令
# 重写后：100万条写命令（无法压缩）
```

## 五、注意事项与常见陷阱

1. **重写是异步的**：不阻塞主线程
2. **重写期间新命令处理**：通过AOF重写缓冲区保证数据完整性
3. **COW开销**：fork时的Copy-On-Write会占用额外内存
4. **不能并行重写**：只能同时进行一次AOF重写
5. **重写触发条件**：auto-aof-rewrite-percentage和auto-aof-rewrite-min-size必须同时满足
6. **RDB+AOF混合模式**：重写时使用RDB格式，更高效
