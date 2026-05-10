# Redis 客户端与编程


## 🔌 Redis 客户端与编程


ioredis (Node.js) 连接池/事件, redis-py (Python) 连接/管道, Spring Data Redis, go-redis 客户端, 连接管理与最佳实践。


## Python redis-py


```
// ========== Python redis-py ==========
// pip install redis

// ========== 基础连接 ==========
import redis

# 简单连接
r = redis.Redis(host='localhost', port=6379, db=0)
r.set('key', 'value')
r.get('key')           # b'value'
r.get('key').decode()  # 'value'

// ========== 连接池 ==========
pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    max_connections=10,
    decode_responses=True  # 自动解码为字符串
)
r = redis.Redis(connection_pool=pool)

// ========== 常用操作 ==========
r.set('name', 'Alice', ex=3600)  # 带过期
r.setnx('lock', '1')               # 不存在才设置

r.mset({'a': 1, 'b': 2})
r.mget(['a', 'b'])

# Hash
r.hset('user:1', 'name', 'Alice')
r.hget('user:1', 'name')
r.hgetall('user:1')  # {'name': 'Alice'}

# List
r.rpush('queue', 'task1')
r.lpop('queue')
r.blpop('queue', timeout=5)

# Set
r.sadd('tags', 'redis', 'python')
r.smembers('tags')
r.sinter('tags1', 'tags2')

# ZSet
r.zadd('leaderboard', {'Alice': 100, 'Bob': 200})
r.zrevrange('leaderboard', 0, 9, withscores=True)

// ========== Pipeline ==========
pipe = r.pipeline()
pipe.set('key1', 'val1')
pipe.set('key2', 'val2')
pipe.incr('counter')
result = pipe.execute()  # [True, True, 1]

// ========== 事务 ==========
pipe = r.pipeline(transaction=True)
pipe.multi()
pipe.set('a', 1)
pipe.set('b', 2)
pipe.execute()

// ========== Lua 脚本 ==========
script = """
    local key = KEYS[1]
    return redis.call('GET', key)
"""
sha = r.script_load(script)
result = r.evalsha(sha, 1, 'mykey')
```


## Node.js ioredis


```
// ========== Node.js ioredis ==========
// npm install ioredis

// ========== 基础连接 ==========
const Redis = require('ioredis');

const redis = new Redis({
    host: 'localhost',
    port: 6379,
    db: 0,
    password: 'optional',
    enableReadyCheck: true,
});

// 连接事件
redis.on('connect', () => console.log('Connected'));
redis.on('ready', () => console.log('Ready'));
redis.on('error', (err) => console.error('Error:', err));
redis.on('close', () => console.log('Connection closed'));

// ========== 常用操作 ==========
await redis.set('name', 'Alice');
await redis.get('name');          // 'Alice'

await redis.setex('session', 3600, 'data');  // 带过期

// Hash
await redis.hset('user:1', 'name', 'Alice', 'age', 28);
await redis.hgetall('user:1');  // { name: 'Alice', age: '28' }

// List
await redis.rpush('queue', 'task1');
await redis.blpop('queue', 0);  // 阻塞弹出

// Set
await redis.sadd('tags', 'redis');
await redis.smembers('tags');

// ZSet
await redis.zadd('leaderboard', 100, 'Alice', 200, 'Bob');
await redis.zrevrange('leaderboard', 0, 9, 'WITHSCORES');

// ========== Pipeline ==========
const pipeline = redis.pipeline();
pipeline.set('key1', 'val1');
pipeline.set('key2', 'val2');
results = await pipeline.exec();

// ========== 事务 ==========
const result = await redis.multi()
    .set('a', 1)
    .set('b', 2)
    .exec();

// ========== Lua 脚本 ==========
redis.defineCommand('getLock', {
    numberOfKeys: 1,
    lua: `return redis.call('SET', KEYS[1], ARGV[1], 'NX', 'EX', ARGV[2])`
});
const ok = await redis.getLock('lock:key', 'uuid', '10');

// ========== 关闭连接 ==========
await redis.quit();  // 等待所有命令完成
await redis.disconnect();  // 立即断开
```


## Java Spring Data Redis


```
// ========== Spring Data Redis ==========

// ========== 依赖 (Maven) ==========

org.springframework.boot
spring-boot-starter-data-redis


// ========== 配置 ==========
# application.yml
spring:
  redis:
    host: localhost
    port: 6379
    password:
    timeout: 2000ms
    lettuce:
      pool:
        max-active: 8
        max-idle: 8
        min-idle: 0

// ========== 使用 RedisTemplate ==========
@Autowired
private RedisTemplate redisTemplate;

// 操作 String
redisTemplate.opsForValue().set("key", "value");
String value = redisTemplate.opsForValue().get("key");

// 操作 Hash
redisTemplate.opsForHash().put("user:1", "name", "Alice");
String name = (String) redisTemplate.opsForHash().get("user:1", "name");

// 操作 List
redisTemplate.opsForList().rightPush("queue", "task1");
String task = redisTemplate.opsForList().leftPop("queue");

// 操作 Set
redisTemplate.opsForSet().add("tags", "redis");
Set tags = redisTemplate.opsForSet().members("tags");

// 操作 ZSet
redisTemplate.opsForZSet().add("leaderboard", "Alice", 100);
Set top = redisTemplate.opsForZSet().reverseRange("leaderboard", 0, 9);

// ========== 过期 ==========
redisTemplate.expire("key", 1, TimeUnit.HOURS);

// ========== @Cacheable 注解 ==========
@Cacheable(value = "users", key = "#id")
public User getUserById(Long id) {
    return userRepository.findById(id).orElse(null);
}
```


## Go go-redis & 通用最佳实践


```
// ========== Go go-redis ==========
// go get github.com/redis/go-redis/v9

import "github.com/redis/go-redis/v9"

// 连接
rdb := redis.NewClient(&redis.Options{
    Addr:     "localhost:6379",
    Password: "",
    DB:       0,
    PoolSize: 10,
})

ctx := context.Background()

// 操作
rdb.Set(ctx, "key", "value", time.Hour)
val, _ := rdb.Get(ctx, "key").Result()

rdb.HSet(ctx, "user:1", "name", "Alice")
rdb.HGetAll(ctx, "user:1").Result()

rdb.RPush(ctx, "queue", "task1")
rdb.BLPop(ctx, 0, "queue")

rdb.ZAdd(ctx, "leaderboard", redis.Z{Score: 100, Member: "Alice"})
rdb.ZRevRangeWithScores(ctx, "leaderboard", 0, 9).Result()

// Pipeline
pipe := rdb.Pipeline()
pipe.Set(ctx, "k1", "v1", 0)
pipe.Set(ctx, "k2", "v2", 0)
pipe.Exec(ctx)

// ========== 通用最佳实践 ==========
// 1. 使用连接池, 不用每次创建连接
// 2. 设置合理的超时 (连接/读写)
// 3. 使用 Pipeline 批量操作
// 4. 使用 Lua 脚本保证原子性
// 5. 异常处理: 重试 + 熔断
// 6. 监控: 连接数/延迟/命中率
// 7. 键命名: app:module:object:id:field
// 8. 序列化: JSON/Protocol Buffers
```


> **Note:** 💡 客户端要点: 使用连接池 (不是每次新建); Pipeline 减少网络; Lua 脚本保证原子; 设置超时和重试; Spring @Cacheable 简化缓存; ioredis Cluster 自动支持集群; 键名加前缀区分应用。


## 练习


<!-- Converted from: 60_Redis 客户端与编程.html -->
