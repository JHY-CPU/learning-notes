# Redis String 与 Key 命令


## 🔤 Redis String 与 Key 命令


String 基本操作 (SET/GET/GETSET)、计数器 (INCR/DECR)、批量操作 (MSET/MGET)、Key 过期 (EXPIRE/TTL)、命名规范与设计。


## String 类型


```
// ========== String (字符串) ==========
// Redis 最基础的数据类型
// 值最大 512MB

// ========== 基本操作 ==========

SET name "Alice"           // 设置键值
GET name                   // 获取: "Alice"
GETSET name "Bob"          // 设置新值并返回旧值: "Alice"

SETNX name "Charlie"       // 不存在才设置 (SET if Not eXists)
                           // 0 (已存在) / 1 (设置成功)

SET name "Dave" NX         // NX = 不存在才设置 (同 SETNX)
SET name "Dave" XX         // XX = 存在才设置 (更新)

MSET k1 v1 k2 v2 k3 v3     // 批量设置
MGET k1 k2 k3              // 批量获取: ["v1", "v2", "v3"]

// ========== 追加与长度 ==========

STRLEN name                // 字符串长度: 4
APPEND name " Smith"       // 追加: "Alice Smith" (返回新长度)

// ========== 子串 ==========

SET msg "Hello, World!"
GETRANGE msg 0 4           // 子串: "Hello"
SETRANGE msg 7 "Redis"     // 替换: "Hello, Redis!"

// ========== 计数器 (原子操作) ==========

SET counter 0
INCR counter                // 1
INCR counter                // 2
INCRBY counter 5            // 7
DECR counter                // 6
DECRBY counter 3            // 3
INCRBYFLOAT price 9.99      // 浮点数: "9.99"

// 计数器应用:
// - 页面访问量: INCR page:article:123
// - 限流: INCR + EXPIRE
// - 序列生成器: INCR id:users

// ========== 超时操作 ==========

SET session:abc "data"
EXPIRE session:abc 3600    // 1 小时后过期
TTL session:abc            // 剩余秒数: 3599

SETEX session:abc 3600 "data"    // SET + EXPIRE 原子操作
PSETEX session:abc 3600000 "data" // 毫秒版

// 取消过期
PERSIST session:abc        // 移除过期时间

// ========== 位图操作 ==========
// 节省内存的位级别操作

SETBIT flags 0 1           // 设置第 0 位为 1
SETBIT flags 1 0           // 设置第 1 位为 0
GETBIT flags 0             // 获取第 0 位: 1
BITCOUNT flags             // 统计 1 的数量
BITOP AND result flags1 flags2  // 位运算
```


## Key 命令


```
// ========== Key 通用命令 ==========

// 通用操作 (所有数据类型通用)

KEYS *               // 所有键 (生产环境别用!)
KEYS user:*          // 匹配模式

SCAN 0 MATCH user:* COUNT 100  // 游标迭代 (生产用)
// SCAN 返回: [next_cursor, [keys]]

EXISTS key           // 是否存在: 0/1
TYPE key             // 类型: string/list/set/zset/hash
TTL key              // 剩余过期秒数: -1=永久, -2=不存在
PTTL key             // 剩余过期毫秒数
OBJECT ENCODING key  // 内部编码: embstr/int/raw

RENAME key newkey     // 重命名
RENAMENX key newkey   // 不存在时才重命名

RANDOMKEY            // 随机返回一个键
SORT list DESC        // 排序 (List/Set/ZSet)

// ========== 删除 ==========
DEL key1 key2        // 删除键: 返回删除数量
UNLINK key1 key2     // 异步删除 (非阻塞)

// ========== 序列化 ==========
DUMP key             // 序列化
RESTORE key 0 data   // 反序列化

// ========== 键命名规范 ==========
// 推荐: object:id:field

// 用户相关
user:1000:name       // 用户 1000 的名字
user:1000:email      // 用户 1000 的邮箱

// 会话
session:abc123       // 会话

// 缓存
cache:article:42     // 缓存文章

// 计数器
page:article:42:views  // 文章访问量

// 限流
rate:api:user:1000   // 用户限流

// 队列
queue:email:send     // 邮件队列

// 优势:
// 1. 按前缀查询 (KEYS user:*)
// 2. 可读性强
// 3. 避免命名冲突
```


## 过期与内存管理


```
// ========== 过期策略 ==========

// Redis 的三种过期删除策略:

// 1. 定期删除 (Active Expiry)
//    每 100ms 随机检查一些过期 key 并删除
//    不是遍历所有 key (性能考虑)

// 2. 惰性删除 (Lazy Expiry)
//    访问 key 时检查是否过期, 过期则删除
//    被动方式

// 3. 内存淘汰 (Eviction)
//    内存满时按策略淘汰 key

// ========== 内存淘汰策略 ==========

// maxmemory-policy 可选值:
// ┌─────────────────┬──────────────────────────┐
// │ 策略             │ 说明                      │
// ├─────────────────┼──────────────────────────┤
// │ noeviction      │ 不淘汰, 写入返回错误      │
// │ allkeys-lru     │ 所有键中淘汰最近最少使用   │
// │ allkeys-lfu     │ 所有键中淘汰最不经常使用  │
// │ volatile-lru    │ 有过期时间的 LRU 淘汰     │
// │ volatile-lfu    │ 有过期时间的 LFU 淘汰     │
// │ volatile-ttl    │ 过期时间最近的淘汰         │
// │ volatile-random │ 有过期时间的随机淘汰      │
// │ allkeys-random  │ 所有键随机淘汰            │
// └─────────────────┴──────────────────────────┘

// 推荐: maxmemory-policy allkeys-lru

// ========== 内存查看 ==========
INFO memory

// 关键指标:
// used_memory: 当前内存使用 (字节)
// used_memory_human: 人类可读格式
// maxmemory: 配置的最大内存
// mem_fragmentation_ratio: 碎片率 (>1.5 需要重启)

// ========== 键数量 ==========
DBSIZE    // 当前数据库键数量
```


## 实际应用示例


```
// ========== 实际应用 ==========

// ========== 1. 缓存用户数据 ==========
SET user:1000 '{"name":"Alice","email":"alice@test.com"}'
EXPIRE user:1000 3600

// 或原子操作:
SETEX user:1000 3600 '{"name":"Alice","email":"alice@test.com"}'

// ========== 2. 计数器 ==========
// 文章阅读量
INCR article:42:views

// 每日活跃用户 (用 Bitmap)
SETBIT active:2024-10-01 user_id 1

// ========== 3. 限流 ==========
// 每秒最多 10 次
INCR rate:api:user:1000
EXPIRE rate:api:user:1000 1
// 检查值是否 > 10

// 或原子操作:
// SET rate:api:user:1000 1 NX EX 1
// INCR rate:api:user:1000
// 然后判断是否 > 10

// ========== 4. 分布式锁 (简单版) ==========
SET lock:resource:123 UUID NX EX 10
// 成功 = 获得锁, 10 秒自动释放
// 解锁: DEL lock:resource:123 (需验证 UUID)
// 注意: 生产用 Redlock 算法

// ========== 5. 自增 ID ==========
INCR id:users          // 1
INCR id:orders         // 1
INCR id:users          // 2
```


> **Note:** 💡 String 要点: 最基础类型, 值最大 512MB; INCR/DECR 原子计数; SETNX 实现分布式锁; SETEX 原子设过期; 键命名 object:id:field; KEYS 生产禁用用 SCAN; 淘汰策略推荐 allkeys-lru。


## 练习


<!-- Converted from: 52_Redis String 与 Key 命令.html -->
