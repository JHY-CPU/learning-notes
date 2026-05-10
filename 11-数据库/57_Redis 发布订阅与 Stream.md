# Redis 发布订阅与 Stream


## 📢 Redis 发布订阅与 Stream


Pub/Sub (SUBSCRIBE/PUBLISH)、模式匹配订阅、Redis Stream 数据结构 (XADD/XREAD/XREADGROUP)、消费者组、消息队列对比。


## 发布订阅 (Pub/Sub)


```
// ========== Pub/Sub ==========
// 发布-订阅模式, 一对多消息广播

// ========== 基本命令 ==========

// 订阅者 1 (阻塞等待消息):
SUBSCRIBE channel:news

// 订阅者 2:
SUBSCRIBE channel:news

// 发布者:
PUBLISH channel:news "Hello, everyone!"
// 返回: (integer) 2 (2 个订阅者收到)

// ========== 模式匹配订阅 ==========

// 订阅匹配模式:
PSUBSCRIBE channel:*            // 订阅所有 channel: 开头的

// 发布:
PUBLISH channel:sports "Match start!"
PUBLISH channel:tech "New AI model!"
// 模式匹配的订阅者都能收到

// ========== 取消订阅 ==========
UNSUBSCRIBE channel:news        // 取消指定
UNSUBSCRIBE                     // 取消所有
PUNSUBSCRIBE channel:*          // 取消模式

// ========== Pub/Sub 命令 ==========

// 查看频道
PUBSUB CHANNELS                 // 活跃的频道
PUBSUB CHANNELS channel:*       // 匹配的频道
PUBSUB NUMSUB channel:news      // 订阅者数量
PUBSUB NUMPAT                   // 模式订阅数量

// ========== 实战: 实时通知 ==========
// 场景: 用户登录通知

// 订阅频道:
SUBSCRIBE user:login

// 登录时发布:
PUBLISH user:login '{"user_id":100,"time":"2024-10-01T10:00:00"}'

// ========== Pub/Sub 限制 ==========
// 1. 消息不持久化 (订阅者不在线 → 消息丢失)
// 2. 无消息确认机制
// 3. 订阅者阻塞不能做其他操作
// 4. 不适合可靠消息传递
// 5. 适合: 实时广播, 日志通知
```


## Redis Stream


```
// ========== Stream (流) ==========
// Redis 5.0+ 引入, 类似 Kafka 的消息队列
// 支持: 持久化, 消费者组, ACK, 回溯消费

// ========== 基本命令 ==========

// 添加消息
XADD mystream * sensor-id 1234 temperature 19.8
// * = 自动生成 ID (时间戳-序号)
// 返回: "1696118400000-0"

XADD mystream MAXLEN 1000 * sensor-id 5678 temperature 20.1
// MAXLEN = 限制流长度 (防止无限增长)

// 指定 ID (不推荐)
XADD mystream 0-1 key value

// ========== 读取消息 ==========

// 从开始读取所有
XRANGE mystream - +                    // - = 开始, + = 结束

// 按时间范围
XRANGE mystream 1696118400000 1696204800000

// 按数量
XRANGE mystream - + COUNT 10

// 反向读取
XREVRANGE mystream + - COUNT 5

// ========== 阻塞读取 ==========

// 从某个 ID 后读取, 阻塞等待
XREAD BLOCK 0 STREAMS mystream 0
// BLOCK 0 = 一直阻塞
// 0 = 从头开始读

// 只读新消息 ($)
XREAD BLOCK 0 STREAMS mystream $

// 每次最多 10 条
XREAD COUNT 10 BLOCK 0 STREAMS mystream $
```


## Stream 消费者组


```
// ========== 消费者组 ==========
// 类似 Kafka 的分区消费者组

// ========== 创建消费者组 ==========
XGROUP CREATE mystream mygroup $ MKSTREAM
// mygroup = 组名
// $ = 只消费新消息
// 0 = 从头消费
// MKSTREAM = 流不存在则创建

// ========== 消费者读取 ==========

// 消费者 1 读取
XREADGROUP GROUP mygroup consumer1
    COUNT 1 BLOCK 2000
    STREAMS mystream >

// 消费者 2 读取 (自动负载均衡)
XREADGROUP GROUP mygroup consumer2
    COUNT 1 BLOCK 2000
    STREAMS mystream >

// > = 只消费未被消费的消息
// 不同消费者消费不同消息 (类似 Kafka partition)

// ========== 消息确认 ==========

// 确认消息 (防止重复)
XACK mystream mygroup "1696118400000-0"

// ========== 查看未确认消息 ==========
XPENDING mystream mygroup
// 显示: 未确认数量, 最早/最晚 ID, 消费者

// 查看某个消费者的未确认:
XPENDING mystream mygroup - + 10 consumer1

// ========== 重新处理失败消息 ==========
XCLAIM mystream mygroup consumer2 60000 "1696118400000-0"
// 将未确认消息转移给其他消费者
// 60000 = 最小空闲时间 (毫秒)

// ========== 管理命令 ==========

// 查看流信息
XLEN mystream                  // 长度

// 查看流详情
XINFO STREAM mystream          // 流信息
XINFO GROUPS mystream          // 消费者组
XINFO CONSUMERS mystream mygroup  // 消费者
```


## 消息队列方案对比


```
// ========== Redis 消息方案对比 ==========
// ┌─────────────┬───────────┬──────────┬───────────┐
// │ 特性         │ List      │ Pub/Sub  │ Stream    │
// ├─────────────┼───────────┼──────────┼───────────┤
// │ 持久化       │ ✅ AOF    │ ❌       │ ✅ AOF    │
// │ ACK 确认     │ ❌        │ ❌       │ ✅        │
// │ 多消费者广播 │ ❌        │ ✅       │ ✅        │
// │ 消费组       │ ❌        │ ❌       │ ✅        │
// │ 回溯消费     │ ❌ pop即删 │ ❌       │ ✅        │
// │ 阻塞读取     │ BRPOP     │ SUBSCRIBE│ XREAD    │
// ├─────────────┼───────────┼──────────┼───────────┤
// │ 适用场景     │ 简单队列  │ 实时广播  │ 可靠消息  │
// └─────────────┴───────────┴──────────┴───────────┘

// ========== 选型建议 ==========
// 简单任务队列: List (LPUSH + BRPOP)
// 实时通知广播: Pub/Sub
// 可靠消息: Stream (消费者组 + ACK)
// 大量消息 (10万+/秒): Kafka (非 Redis)

// ========== Stream vs Kafka ==========
// Similarities:
// - 持久化消息
// - 消费者组 + 负载均衡
// - 消息 offset
// - 回溯消费
//
// Redis Stream 优势:
// - 简单 (不需要搭建 Kafka 集群)
// - 低延迟 (内存)
// - 集成到 Redis 生态
//
// Kafka 优势:
// - 超高吞吐 (磁盘顺序读写)
// - 长期存储 (不限制内存)
// - 分区内严格有序
// - 生态系统完善
```


> **Note:** 💡 Pub/Sub: 实时广播, 消息不持久, 订阅者离线丢失。Stream: 持久化消息队列, 支持消费者组和 ACK, 类似 Kafka 但轻量。选型: 简单队列用 List; 实时广播 Pub/Sub; 可靠消息 Stream; 海量消息 Kafka。


## 练习


<!-- Converted from: 57_Redis 发布订阅与 Stream.html -->
