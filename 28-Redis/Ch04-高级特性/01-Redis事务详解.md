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
