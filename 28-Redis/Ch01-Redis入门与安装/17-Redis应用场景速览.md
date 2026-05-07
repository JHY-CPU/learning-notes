# Redis应用场景速览

## 一、概念说明

Redis因其高性能和丰富的数据类型，被广泛应用于各种场景。了解常见应用场景有助于正确使用Redis。

## 二、核心应用场景

### 缓存

```bash
# 最常见的使用场景
# 缓存数据库查询结果

# 设置缓存（带过期时间）
SET user:1001 '{"name":"张三","age":25}' EX 3600

# 获取缓存
GET user:1001
# 输出: "{\"name\":\"张三\",\"age\":25}"

# 缓存不存在时查询数据库
GET user:1002
# 输出: (nil)
# -> 查询数据库 -> SET user:1002 data EX 3600
```

### 会话存储

```bash
# 分布式Session共享
# 存储用户Session信息

HSET session:abc123 user_id 1001
HSET session:abc123 username "张三"
HSET session:abc123 login_time 1700000000
EXPIRE session:abc123 1800

# 验证Session
HGETALL session:abc123
# 输出: 1) "user_id" 2) "1001"
#       3) "username" 4) "张三"
#       5) "login_time" 6) "1700000000"
```

### 排行榜

```bash
# 使用有序集合实现实时排行榜

ZADD leaderboard 100 "player1"
ZADD leaderboard 200 "player2"
ZADD leaderboard 150 "player3"

# 获取前3名
ZREVRANGE leaderboard 0 2 WITHSCORES
# 输出: 1) "player2" 2) "200"
#       3) "player3" 4) "150"
#       5) "player1" 6) "100"

# 增加分数
ZINCRBY leaderboard 50 "player1"
```

### 消息队列

```bash
# 使用List实现简单消息队列

# 生产者
LPUSH queue:orders '{"order_id":1001,"amount":99.9}'

# 消费者（阻塞式）
BRPOP queue:orders 0
# 输出: 1) "queue:orders"
#       2) "{\"order_id\":1001,\"amount\":99.9}"
```

### 分布式锁

```bash
# 使用SET NX EX实现分布式锁

# 获取锁
SET lock:resource "uuid-value" NX EX 30
# 输出: OK（获取成功）
# 输出: (nil)（已被占用）

# 释放锁（使用Lua脚本保证原子性）
EVAL "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end" 1 lock:resource "uuid-value"
```

### 计数器

```bash
# 文章阅读量计数器

INCR article:1001:views
# 输出: (integer) 1

INCR article:1001:views
# 输出: (integer) 2

# 获取当前阅读量
GET article:1001:views
# 输出: "2"
```

### 限流

```bash
# 固定窗口限流（每分钟100次）

INCR rate:limit:user1001
EXPIRE rate:limit:user1001 60

# 检查是否超限
GET rate:limit:user1001
# 如果 > 100，拒绝请求
```

### 地理位置

```bash
# 存储地理位置信息
GEOADD stores 116.40 39.90 "store1"
GEOADD stores 116.41 39.91 "store2"

# 查找附近1km内的店铺
GEORADIUS stores 116.40 39.90 1 km
# 输出: 1) "store1"
#       2) "store2"
```

## 三、注意事项与常见陷阱

1. **缓存雪崩**：大量Key同时过期导致数据库压力骤增
2. **缓存穿透**：查询不存在的数据，每次都穿透到数据库
3. **缓存击穿**：热点Key过期瞬间大量请求打到数据库
4. **数据一致性**：缓存和数据库之间的一致性问题
5. **内存容量**：合理设置过期时间，避免内存溢出
6. **序列化选择**：选择高效的序列化方式减少内存使用
