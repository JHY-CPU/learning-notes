# Redis事务基础

## 一、概念说明

Redis事务允许将多个命令打包，按顺序执行。事务中的命令要么全部执行，要么全部不执行。Redis事务通过MULTI/EXEC/WATCH命令实现。

## 二、具体用法

### MULTI/EXEC基本用法

```bash
# 开始事务
MULTI
# 输出: OK

# 事务中的命令会入队
SET account:1 100
# 输出: QUEUED

SET account:2 50
# 输出: QUEUED

# 执行事务
EXEC
# 输出: 1) OK 2) OK

# 放弃事务
MULTI
SET key1 "value1"
DISCARD
# 输出: OK
GET key1
# 输出: (nil)（事务已取消）
```

### WATCH乐观锁

```bash
# 模拟转账操作
SET balance:1 100
SET balance:2 50

# 监视账户余额
WATCH balance:1
# 输出: OK

# 检查余额后执行转账
MULTI
DECRBY balance:1 30
INCRBY balance:2 30
EXEC
# 如果balance:1在WATCH和EXEC之间被修改
# EXEC返回(nil)，事务失败

# 检查事务结果
GET balance:1
# 输出: "70"
GET balance:2
# 输出: "80"
```

### 事务中的错误处理

```bash
# 语法错误（命令入队时检测）
MULTI
SET key "value"
INCR key    # key不是整数，但命令入队时不报错
EXEC
# 输出: 1) OK
#        2) (error) ERR value is an integer or out of range
# 事务中部分命令失败，但其他命令仍会执行

# 取消WATCH
UNWATCH
# 输出: OK
```

## 三、Redis事务 vs 数据库事务

| 特性 | Redis事务 | 数据库事务 |
|------|-----------|------------|
| 原子性 | 部分支持 | 完全支持 |
| 回滚 | 不支持 | 支持 |
| 隔离级别 | 无 | 支持多种 |
| 一致性 | 命令级别 | ACID |
| 锁机制 | WATCH乐观锁 | 悲观锁/乐观锁 |

## 四、注意事项与常见陷阱

1. **Redis事务不支持回滚**：命令执行错误后，其他命令仍会执行
2. **WATCH是乐观锁**：检查键是否被修改，被修改则事务失败
3. **MULTI/EXEC之间不执行命令**：命令只是入队
4. **事务中的命令是顺序执行**：不是并行执行
5. **WATCH在EXEC后自动取消**：不需要手动UNWATCH
6. **Lua脚本替代事务**：复杂操作建议使用Lua脚本保证原子性
