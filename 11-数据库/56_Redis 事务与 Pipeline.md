# Redis 事务与 Pipeline


## 📦 Redis 事务与 Pipeline


MULTI/EXEC 事务、WATCH 乐观锁、Pipeline 批量请求优化、事务与 Pipeline 对比、Redis 事务 vs SQL 事务差异。


## Redis 事务


```
// ========== Redis 事务 ==========
// MULTI / EXEC / DISCARD / WATCH

// ========== 事务流程 ==========
// 1. MULTI — 开始事务
// 2. 命令入队 (不立即执行)
// 3. EXEC — 依次执行所有命令

> MULTI
OK
> SET balance 100
QUEUED
> INCR balance
QUEUED
> DECR balance
QUEUED
> EXEC
1) OK
2) (integer) 101
3) (integer) 100

// ========== 取消事务 ==========
> MULTI
> SET key1 "value1"
> DISCARD           // 取消事务, 清空队列
// 所有命令不执行

// ========== 事务特性 ==========
// Redis 事务特点:
// 1. 所有命令按顺序执行 (独占)
// 2. 不会被其他客户端的命令打断
// 3. 不支持回滚!
//    语法错误 → 全部不执行
//    运行时错误 → 其他命令继续执行

// ========== 语法错误 (全部失败) ==========
> MULTI
> SET key "value"
> SET key                    // 语法错误!
> EXEC
(error) EXECABORTED          // 事务取消

// ========== 运行时错误 (部分成功) ==========
> MULTI
> SET key "hello"
> INCR key                   // INCR 字符串 → 运行时错误
> SET other "world"
> EXEC
1) OK
2) (error) WRONGTYPE         // INCR 失败
3) OK                        // 但 SET other 成功了!
// Redis 不回滚!
```


## WATCH 乐观锁


```
// ========== WATCH 乐观锁 ==========
// 用于实现 CAS (Check-And-Set)
// WATCH 监视 key, 如果 key 被修改, 事务失败

// ========== 示例: 转账 ==========

// Alice 有 100 元, Bob 有 50 元
// Alice 转 20 元给 Bob

> WATCH balance:alice              // 监视 Alice 余额
> GET balance:alice
"100"
> MULTI
> DECRBY balance:alice 20          // Alice -20
> INCRBY balance:bob 20            // Bob +20
> EXEC
1) (integer) 80                    // 成功

// 如果 EXEC 前有其他客户端修改了 balance:alice:
// EXEC → (nil) (返回 nil, 事务失败)
// 应用需要重试

// ========== 完整重试模式 ==========

// Lua 伪代码:
function transfer(from, to, amount)
    while true do
        redis.call("WATCH", from)
        local bal = redis.call("GET", from)
        if tonumber(bal) < amount then
            redis.call("UNWATCH")
            return error("余额不足")
        end
        local ok = redis.call("MULTI")
        redis.call("DECRBY", from, amount)
        redis.call("INCRBY", to, amount)
        local result = redis.call("EXEC")
        if result then
            return "ok"           // 成功
        end
        // 失败 → 循环重试
    end
end

// ========== UNWATCH ==========
// 取消监视 (事务执行/取消后自动取消)

UNWATCH  // 手动取消
```


## Pipeline 管道


```
// ========== Pipeline (管道) ==========
// 将多个命令批量发送给服务器
// 减少网络往返 (RTT, Round Trip Time)

// ========== 原理 ==========
// 无 Pipeline: 每次命令等待响应
//   SET a 1 → 等待 → GET a → 等待 → SET b 2 → 等待
//
// 有 Pipeline: 一次发送多个命令
//   [SET a 1, GET a, SET b 2] → 一次等待 → [OK, "1", OK]

// 性能提升: 批量 100 个命令可提升 10x-100x

// ========== redis-cli Pipeline ==========
# 创建命令文件
echo -e "SET k1 v1\nSET k2 v2\nGET k1" > commands.txt
cat commands.txt | redis-cli --pipe

# 或:
printf 'SET k1 v1\r\nSET k2 v2\r\nGET k1\r\n' | redis-cli --pipe

// ========== Python Pipeline ==========
import redis
r = redis.Redis()

# 不启用 Pipeline (1000 次网络往返)
for i in range(1000):
    r.set(f'key:{i}', i)

# 启用 Pipeline (1 次网络往返 + 1000 次命令)
pipe = r.pipeline()
for i in range(1000):
    pipe.set(f'key:{i}', i)
pipe.execute()

// ========== Node.js Pipeline ==========
const Redis = require('ioredis');
const redis = new Redis();

// 无 Pipeline
for (let i = 0; i < 1000; i++) {
    await redis.set(`key:${i}`, i);
}

// 有 Pipeline
const pipeline = redis.pipeline();
for (let i = 0; i < 1000; i++) {
    pipeline.set(`key:${i}`, i);
}
await pipeline.exec();
```


## 事务 vs Pipeline 对比


```
// ========== 事务 vs Pipeline 对比 ==========
// ┌──────────────┬─────────────────┬──────────────────┐
// │ 特性          │ 事务 (MULTI)    │ Pipeline         │
// ├──────────────┼─────────────────┼──────────────────┤
// │ 原子性       │ ✅              │ ❌               │
// │ 隔离性       │ ✅ (独占)       │ ❌ (可能交错)    │
// │ 减少 RTT     │ ✅              │ ✅               │
// │ 回滚         │ ❌ (不支持)     │ ❌               │
// │ 立即执行     │ ❌ (EXEC 才执)  │ 立即 (发送就执)  │
// │ 适用场景     │ 需要原子性      │ 批量操作, 性能   │
// └──────────────┴─────────────────┴──────────────────┘

// ========== 实战组合: Pipeline + 事务 ==========
// 用 Pipeline 包装事务, 兼顾性能和原子性

// Python:
pipe = r.pipeline(transaction=True)  // 事务模式
pipe.multi()                          // 相当于 MULTI
pipe.set('a', 1)
pipe.set('b', 2)
pipe.execute()                        // 相当于 EXEC

// Node.js (ioredis):
redis.multi()
    .set('a', 1)
    .set('b', 2)
    .exec()                           // 原子执行

// ========== Lua 脚本 vs 事务 ==========
// Lua 脚本是更好的选择:
// 1. 原子执行
// 2. 可编程 (逻辑判断)
// 3. 可回滚
// 4. 减少网络次数

// 推荐优先级:
// 简单批量 → Pipeline
// 需要原子 → Lua 脚本
// 需要乐观锁 → WATCH + MULTI
```


> **Note:** 💡 事务要点: MULTI 开始, EXEC 执行; 不支持回滚 (注意); WATCH 实现乐观锁; 事务保证隔离但不保证回滚。Pipeline 要点: 批量减少网络往返; 不保证原子性; 可用 Pipeline+事务组合; Lua 脚本替代事务更灵活。


## 练习


<!-- Converted from: 56_Redis 事务与 Pipeline.html -->
