# Redis事务详解

## 一、概念说明

Redis事务允许将多个命令打包，按顺序原子性执行。通过MULTI/EXEC/WATCH命令实现。与数据库事务不同，Redis事务不支持回滚。

## 二、具体用法

### MULTI/EXEC/DISCARD

```bash
# 开始事务
MULTI
# 输出: OK

# 事务中的命令入队
SET account:1 100
# 输出: QUEUED
SET account:2 50
# 输出: QUEUED

# 执行事务
EXEC
# 输出: 1) OK 2) OK

# 放弃事务
MULTI
SET key1 "val1"
DISCARD
# 输出: OK
```

### WATCH乐观锁

```bash
# 监视键
WATCH balance:1
# 如果balance:1在EXEC前被修改，事务失败

MULTI
DECRBY balance:1 30
INCRBY balance:2 30
EXEC
# 如果被修改，返回(nil)

# 取消监视
UNWATCH
```

## 三、错误处理

```bash
# 入队错误（语法错误）
MULTI
SET key "value"
INCR key  # key不是整数，但入队时不报错
EXEC
# 部分成功：1) OK 2) (error)
# 其他命令仍会执行

# Redis事务不支持回滚
# 设计理念：命令错误是编程错误，应在开发阶段发现
```

## 四、注意事项

1. 不支持回滚：命令错误后其他命令仍执行
2. WATCH是乐观锁：检查键是否被修改
3. 原子性：事务中的命令要么全执行，要么全不执行
4. Lua脚本替代：复杂原子操作使用Lua脚本

## 五、WATCH机制深入

```bash
# WATCH的CAS（Compare-And-Set）模式
# 场景：转账操作

# 客户端A
WATCH balance:1
GET balance:1
# 返回 100

MULTI
DECRBY balance:1 30
INCRBY balance:2 30
EXEC
# 如果balance:1在此期间被修改，EXEC返回(nil)
# 需要在应用层重试

# Python实现CAS
def transfer(r, from_key, to_key, amount):
    with r.pipeline() as pipe:
        while True:
            try:
                pipe.watch(from_key)
                balance = int(pipe.get(from_key))
                if balance < amount:
                    pipe.unwatch()
                    raise Exception("余额不足")

                pipe.multi()
                pipe.decrby(from_key, amount)
                pipe.incrby(to_key, amount)
                pipe.execute()
                break
            except redis.WatchError:
                # 键被修改，重试
                continue
```

## 六、事务 vs Lua脚本

```bash
# 事务特点
# 1. 命令入队时只检查语法，不检查数据类型
# 2. 执行时部分失败不影响其他命令
# 3. 不支持回滚
# 4. WATCH实现乐观锁

# Lua脚本特点
# 1. 原子执行，不被其他命令打断
# 2. 可以包含复杂逻辑判断
# 3. 支持条件执行
# 4. EVALSHA减少网络传输

# 选择建议
# 简单原子操作 → 事务
# 复杂逻辑判断 → Lua脚本
# 需要回滚 → Lua脚本
# 批量操作 → Pipeline + 事务
```

## 七、事务性能考量

```bash
# 事务中的命令不会被其他客户端命令打断
# 但事务执行期间会阻塞Redis

# 性能建议
# 1. 事务中的命令尽量少
# 2. 避免在事务中执行慢操作
# 3. 大Key操作不要放在事务中
# 4. 使用Pipeline减少RTT开销

# 监控事务执行
MULTI
SET key1 val1
SET key2 val2
EXEC
# 通过SLOWLOG查看是否慢
SLOWLOG GET 10
```
