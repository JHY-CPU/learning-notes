# 02-调优实战

## JVM 调优

### JVM 内存结构

JVM 运行时内存分为以下几个区域：

- **堆（Heap）**：存放对象实例，是 GC 管理的主要区域。通过 `-Xms` 和 `-Xmx` 设置初始和最大大小。
- **栈（Stack）**：每个线程私有，存储局部变量、方法调用栈帧。`-Xss` 设置线程栈大小（默认 512KB-1MB）。
- **方法区/元空间（Metaspace）**：存储类元数据、常量池。JDK 8+ 使用本地内存（`-XX:MaxMetaspaceSize`）。
- **直接内存（Direct Memory）**：NIO 的 `DirectByteBuffer` 使用，不受 GC 管理但可触发 Full GC。`-XX:MaxDirectMemorySize` 限制。
- **本地方法栈**：为 native 方法服务。

堆内部分代：新生代（Eden + Survivor0 + Survivor1）和老年代。默认比例 `-XX:NewRatio=2`（老年代:新生代 = 2:1）。

### GC 算法

| GC 算器 | 特点 | 适用场景 |
|---------|------|----------|
| **Serial** | 单线程，Stop-The-World | 小型应用、客户端程序 |
| **Parallel (Throughput)** | 多线程并行回收，注重吞吐量 | 批处理、后台计算任务 |
| **CMS** | 并发标记清除，低延迟（已在 JDK 14 移除） | 已不推荐使用 |
| **G1** | 分 Region 管理，可预测停顿时间 | JDK 9+ 默认，通用场景 |
| **ZGC** | 着色指针 + 读屏障，亚毫秒级停顿 | 大堆（TB 级）、低延迟要求 |
| **Shenandoah** | 并发压缩，与堆大小无关的停顿时间 | 低延迟、大堆场景 |

### GC 日志分析

- JDK 9+ 统一日志参数：`-Xlog:gc*:file=gc.log:time,uptime,level,tags`
- JDK 8 参数：`-XX:+PrintGCDetails -XX:+PrintGCDateStamps -Xloggc:gc.log`
- 关键指标：GC 频率、GC 耗时、GC 前后堆大小变化、晋升失败（promotion failure）。
- 分析工具：**GCViewer**（图形化展示）、**gceasy.io**（在线分析）、**HP jmeter**。

### JVM 参数调优

```
# 堆大小设置（生产环境建议 Xms = Xmx，避免动态扩缩）
-Xms4g -Xmx4g

# 新生代大小
-Xmn1g                    # 固定新生代大小
-XX:NewRatio=2            # 老年代:新生代 = 2:1

# G1 GC 调优
-XX:+UseG1GC
-XX:MaxGCPauseMillis=200  # 目标最大停顿时间
-XX:G1HeapRegionSize=16m  # Region 大小

# 元空间
-XX:MetaspaceSize=256m
-XX:MaxMetaspaceSize=512m

# 其他
-XX:+HeapDumpOnOutOfMemoryError
-XX:HeapDumpPath=/logs/heapdump.hprof
-XX:+DisableExplicitGC     # 禁止 System.gc()
```

### 内存泄漏排查

1. **现象识别**：堆内存持续增长、频繁 Full GC、OOM。
2. **heap dump 采集**：`jmap -dump:format=b,file=heap.hprof <pid>` 或自动触发 OOM dump。
3. **MAT（Memory Analyzer Tool）分析**：
   - Leak Suspects Report 自动检测泄漏嫌疑对象。
   - Dominator Tree 查看大对象及其引用链。
   - Histogram 对比两次 dump 的对象数量变化。
4. **常见泄漏模式**：静态集合未清理、缓存无淘汰策略、监听器未注销、ThreadLocal 未 remove、类加载器泄漏。

### 线程 Dump 分析

- 采集：`jstack <pid>` 或 `jcmd <pid> Thread.print`。
- 分析内容：
  - **死锁检测**：查找 "Found one Java-level deadlock"。
  - **锁竞争**：大量线程处于 `BLOCKED` 状态，持有同一把锁。
  - **线程池耗尽**：大量线程在等待任务队列。
  - **I/O 阻塞**：线程处于 `RUNNABLE` 但堆顶在 native I/O 调用。
- 连续采集 3-5 次 dump（间隔 5-10 秒），对比分析线程状态变化。

---

## 操作系统调优

### 文件描述符限制

```bash
# 查看当前限制
ulimit -n                # 单进程限制
cat /proc/sys/fs/file-max  # 系统全局限制

# 临时修改
ulimit -n 65535

# 永久修改 /etc/security/limits.conf
* soft nofile 65535
* hard nofile 65535
```

每个 TCP 连接占用一个文件描述符，高并发服务器必须提升此限制。

### TCP 参数优化

```bash
# 缓冲区
net.core.rmem_max = 16777216       # 接收缓冲区最大值
net.core.wmem_max = 16777216       # 发送缓冲区最大值
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# TIME_WAIT 优化
net.ipv4.tcp_tw_reuse = 1          # 允许复用 TIME_WAIT 连接
net.ipv4.tcp_fin_timeout = 30      # FIN_WAIT2 超时时间
net.ipv4.tcp_max_tw_buckets = 5000 # 最大 TIME_WAIT 数量

# 连接数
net.core.somaxconn = 65535         # listen 队列最大长度
net.ipv4.tcp_max_syn_backlog = 65535
net.core.netdev_max_backlog = 65535
```

### 内存管理

- **Swap**：生产环境建议 `vm.swappiness=1` 或 `vm.swappiness=0`（数据库场景禁用 Swap）。
- **NUMA**：多路服务器的 NUMA 架构影响内存访问延迟，使用 `numactl --interleave` 或绑定 CPU 亲和性。
- **大页（Huge Pages）**：减少 TLB miss，适合大内存应用（如数据库、JVM 使用 `-XX:+UseLargePages`）。

### I/O 调度器

- `deadline`：适合数据库场景，保证 I/O 请求在截止时间内完成。
- `noop`：适合 SSD 和虚拟化环境（无旋转寻道开销）。
- `mq-deadline`：多队列版本，现代内核推荐。
- 查看/修改：`cat /sys/block/sda/queue/scheduler`。

---

## 数据库调优

### 慢查询分析（EXPLAIN）

```sql
EXPLAIN SELECT * FROM orders WHERE user_id = 123 AND status = 'pending';
```

关注字段：
- **type**：`ALL`（全表扫描）> `index` > `range` > `ref` > `eq_ref` > `const`。
- **key**：实际使用的索引，`NULL` 表示未使用索引。
- **rows**：预估扫描行数，越少越好。
- **Extra**：`Using filesort`（需要排序）、`Using temporary`（使用临时表）是需要关注的信号。

### 索引优化

- **覆盖索引**：查询的所有字段都在索引中，无需回表。`SELECT id, name FROM users WHERE age > 20`，若 `(age, id, name)` 为联合索引则覆盖。
- **联合索引**：遵循最左前缀原则，区分度高的字段在前。
- **索引下推（ICP）**：MySQL 5.6+，在存储引擎层过滤非索引字段，减少回表次数。
- **避免索引失效**：不在索引列上使用函数、避免隐式类型转换、LIKE 左模糊不走索引。

### 查询优化器

- **统计信息**：定期 `ANALYZE TABLE` 更新统计信息，帮助优化器选择最优执行计划。
- **索引提示**：`USE INDEX / FORCE INDEX` 强制使用特定索引（作为最后手段）。
- **子查询优化**：MySQL 5.6+ 对 `IN (subquery)` 自动优化为 semi-join。

### 连接池配置

- **HikariCP**（推荐）：轻量、高性能，关键参数：
  - `maximumPoolSize`：根据公式 `核心数 × 2 + 磁盘数` 估算。
  - `minimumIdle`：最小空闲连接数。
  - `connectionTimeout`：获取连接的超时时间（默认 30s）。
  - `maxLifetime`：连接最大存活时间（默认 30 分钟，应小于数据库的 wait_timeout）。

### 读写分离与分库分表

- **读写分离**：主库处理写操作，从库处理读操作。使用中间件（ShardingSphere、MyCat）或应用层路由。
- **分库分表**：
  - 水平分片：按 user_id 取模、按时间范围、一致性哈希。
  - 垂直分片：按业务域拆分数据库。
  - 分片键选择是核心设计决策，影响查询路由效率。

---

## 网络调优

### 连接池

- HTTP 连接池：`OkHttp`、`Apache HttpClient` 配置连接池复用 TCP 连接。
- gRPC 连接池：`ManagedChannel` 内置连接复用，配置 `maxConcurrentStreams`。
- 数据库连接池：HikariCP、Druid，避免频繁创建销毁连接的开销。

### HTTP/2 多路复用

- 单个 TCP 连接上并发多个请求/响应，消除 HTTP/1.1 的队头阻塞。
- 服务器推送（Server Push）、头部压缩（HPACK）。
- 适用于微服务间通信、API Gateway。

### gRPC 与 Protobuf

- gRPC 基于 HTTP/2，支持双向流、多路复用。
- Protobuf 二进制序列化，比 JSON 小 3-10 倍，序列化/反序列化快 5-10 倍。
- 适合内部微服务间高性能通信。

### CDN 与边缘缓存

- 静态资源（图片、JS、CSS）部署到 CDN，减少源站压力。
- 边缘计算（Cloudflare Workers、AWS Lambda@Edge）将计算下沉到用户最近的节点。
- 缓存策略：`Cache-Control`、`ETag`、`Last-Modified`、`stale-while-revalidate`。

---

## 应用层调优

### 异步化

- **CompletableFuture**：Java 异步编程，支持链式编排、异常处理、超时控制。
  ```java
  CompletableFuture.supplyAsync(() -> queryUser(id), executor)
      .thenCompose(user -> queryOrders(user.getId()))
      .thenApply(orders -> enrichOrders(orders))
      .exceptionally(ex -> fallback(ex));
  ```
- **响应式编程**：Project Reactor、RxJava，基于背压的异步数据流处理。
- **异步优势**：释放线程、提高吞吐量、降低延迟（并行调用多个服务）。

### 批处理与聚合

- 数据库批量 INSERT：减少网络往返，`INSERT INTO ... VALUES (...), (...), (...)`。
- 消息批量消费：Kafka `max.poll.records`、RocketMQ 批量拉取。
- 请求聚合：BFF 层合并多个微服务调用，减少客户端请求次数。

### 数据结构与算法优化

- 选择合适的数据结构：`HashMap` vs `TreeMap`、`ArrayList` vs `LinkedList`。
- 使用对象池减少 GC 压力（如 Netty 的 `Recycler`）。
- 避免不必要的装箱/拆箱（`int` vs `Integer`）。
- 字符串拼接使用 `StringBuilder` 而非 `+`。

### 减少序列化/反序列化开销

- 内部服务调用使用 Protobuf、Kryo、Hessian 替代 JSON。
- 缓存序列化结果避免重复计算。
- 使用 `ThreadLocal` 缓存序列化器/反序列化器实例（如 Jackson 的 `ObjectMapper`）。
- 考虑零拷贝技术（`sendfile`、`mmap`）减少数据在内核态与用户态之间的复制。
