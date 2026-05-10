# Redis Lua 脚本


## 📜 Redis Lua 脚本


EVAL/EVALSHA 命令、Lua 脚本编写、原子操作实现 (分布式锁/限流/库存扣减)、脚本缓存、Redis 函数 (7.0+)。


## Lua 脚本基础


```
// ========== Redis Lua 脚本 ==========
// Lua 脚本在 Redis 中原子执行
// 不会被打断, 所有命令原子完成

// ========== EVAL ==========
// EVAL script numkeys key1 key2 ... arg1 arg2 ...

// Hello World:
EVAL "return 'Hello Redis'" 0
// "Hello Redis"

// 访问 KEYS 和 ARGV:
EVAL "return {KEYS[1], KEYS[2], ARGV[1]}" 2 key1 key2 arg1
// 1) "key1"
// 2) "key2"
// 3) "arg1"

// ========== Lua 与 Redis 交互 ==========

// redis.call() — 方法调用, 错误时抛出异常
EVAL "return redis.call('SET', KEYS[1], ARGV[1])" 1 mykey "hello"

// redis.pcall() — 调用, 错误时返回错误对象 (不抛异常)
EVAL "return redis.pcall('SET', KEYS[1], ARGV[1])" 1 mykey "hello"

// ========== Lua 类型和 Redis 类型 ==========
// Lua nil → Redis nil
// Lua number → Redis integer (浮点截断)
// Lua table (数组) → Redis multi bulk reply
// Lua table (键值) → Redis status/map reply

// ========== 常用 Lua 函数 ==========
// string.len(s), string.sub(s, i, j)
// table.insert(t, v), table.remove(t, i)
// tonumber(s), tostring(n)
// #table - 表长度
```


## 实战脚本


```
// ========== 1. 原子分布式锁 ==========

// 加锁脚本:
local key = KEYS[1]
local uuid = ARGV[1]
local ttl = tonumber(ARGV[2])

-- SET key uuid NX EX ttl
local result = redis.call('SET', key, uuid, 'NX', 'EX', ttl)
return result  -- OK / nil

// 解锁脚本 (验证身份再删除):
local key = KEYS[1]
local uuid = ARGV[1]

local value = redis.call('GET', key)
if value == uuid then
    redis.call('DEL', key)
    return 1  -- 解锁成功
end
return 0      -- 无权解锁

// ========== 2. 限流 (滑动窗口) ==========

local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])  -- 时间窗口 (毫秒)
local limit = tonumber(ARGV[3])   -- 最大次数

-- 移除窗口外的请求
redis.call('ZREMRANGEBYSCORE', key, 0, now - window)

-- 当前窗口请求数
local count = redis.call('ZCARD', key)

if count >= limit then
    return 0  -- 限流
end

-- 记录请求
redis.call('ZADD', key, now, now .. ':' .. math.random())
redis.call('EXPIRE', key, window / 1000 + 1)

return 1  -- 放行

// ========== 3. 原子扣库存 ==========

local key = KEYS[1]       -- 库存 key (Hash)
local qty = tonumber(ARGV[1])  -- 扣减数量

local stock = redis.call('HGET', key, 'stock') or 0

if tonumber(stock) < qty then
    return -1  -- 库存不足
end

redis.call('HINCRBY', key, 'stock', -qty)
redis.call('HINCRBY', key, 'sold', qty)

return redis.call('HGET', key, 'stock')  -- 返回剩余库存
```


## 脚本管理与优化


```
// ========== EVALSHA (脚本缓存) ==========
// Redis 会缓存已执行过的脚本

EVAL "return 'hello'" 0  // 首次执行, 传输代码

// 脚本会被缓存, 返回 SHA1 摘要
// 之后可以用 EVALSHA:

EVALSHA "sha1_hex" 0     // 只传 SHA1, 无需传代码

// ========== SCRIPT 命令 ==========

SCRIPT LOAD "return 'hello'"
// "sha1_hex" — 缓存脚本, 返回 SHA1

SCRIPT EXISTS "sha1_hex"
// 1 — 脚本是否存在

SCRIPT FLUSH
// 清除所有缓存脚本

SCRIPT KILL
// 终止正在执行的脚本 (如果未执行写操作)

// ========== 脚本超时 ==========
// 默认 Lua 脚本最大执行 5 秒
// lua-time-limit 5000 (redis.conf)

// 如果脚本超时:
// 1. 脚本未执行写 → SCRIPT KILL
// 2. 脚本已执行写 → SHUTDOWN NOSAVE (重启)

// ========== 性能建议 ==========
// 1. 脚本尽量短 (几毫秒内完成)
// 2. 不要在脚本中做耗时操作
// 3. 优先使用 EVALSHA (减少网络传输)
// 4. 脚本中尽量用 KEYS 数组 (非硬编码)
// 5. 使用 redis-logic 声明周期管理脚本

// ========== Redis 7.0+ 函数 ==========
// 比脚本更高级, 可管理

// FCALL 调用函数 (类似存储过程)
// FUNCTION LOAD 加载函数
// FUNCTION DELETE 删除函数
```


## 完整示例代码


```
// ========== 应用层 Lua 脚本管理 ==========

// ========== Python ==========
import redis

r = redis.Redis()

# 加载脚本
lock_script = """
    local key = KEYS[1]
    local uuid = ARGV[1]
    local ttl = tonumber(ARGV[2])
    return redis.call('SET', key, uuid, 'NX', 'EX', ttl)
"""

# 加载并获取 SHA
sha = r.script_load(lock_script)

# 执行脚本 (自动用 EVALSHA, 失败回退 EVAL)
result = r.evalsha(sha, 1, 'lock:resource', 'uuid-123', '10')

# 或直接用 eval:
result = r.eval(lock_script, 1, 'lock:resource', 'uuid-123', '10')

// ========== Node.js (ioredis) ==========
const Redis = require('ioredis');
const redis = new Redis();

// 定义脚本
const lockScript = `
    local key = KEYS[1]
    local uuid = ARGV[1]
    local ttl = tonumber(ARGV[2])
    return redis.call('SET', key, uuid, 'NX', 'EX', ttl)
`;

// 加载并执行
redis.defineCommand('acquireLock', {
    numberOfKeys: 1,
    lua: lockScript
});

// 调用 (类似普通命令)
const result = await redis.acquireLock('lock:resource', 'uuid-123', '10');

// 或直接用 eval:
await redis.eval(lockScript, 1, 'lock:resource', 'uuid-123', '10');
```


> **Note:** 💡 Lua 要点: EVAL 传脚本和参数; redis.call() 调用 Redis 命令; 原子执行 (不会被打断); 适合分布式锁/限流/扣库存; EVALSHA 减少网络传输; 脚本保持简短 (< 5 秒); 7.0+ 可以用函数。


## 练习


<!-- Converted from: 59_Redis Lua 脚本.html -->
