# Redis基本命令

## 一、概念说明

Redis基本命令是最常用的操作，包括SET/GET/DEL/EXISTS/EXPIRE等。这些命令是Redis操作的基础，需要熟练掌握。

## 二、具体用法

### SET与GET

```bash
# 设置键值
SET name "Redis"
# 输出: OK

# 获取值
GET name
# 输出: "Redis"

# 设置带过期时间的键（单位：秒）
SET session:token "abc123" EX 3600
# 输出: OK

# 设置键（仅当键不存在时）
SET lock:resource "locked" NX EX 10
# 输出: OK（首次设置成功）
# 输出: (nil)（键已存在时返回nil）

# 设置键（仅当键存在时）
SET name "NewRedis" XX
# 输出: OK（键存在时修改成功）

# 批量设置
MSET key1 "value1" key2 "value2" key3 "value3"
# 输出: OK

# 批量获取
MGET key1 key2 key3
# 输出: 1) "value1" 2) "value2" 3) "value3"
```

### DEL与EXISTS

```bash
# 删除键
DEL name
# 输出: (integer) 1（成功删除1个）

DEL key1 key2 key3
# 输出: (integer) 3

# 检查键是否存在
EXISTS name
# 输出: (integer) 1（存在）
EXISTS nonexistent
# 输出: (integer) 0（不存在）
```

### EXPIRE与TTL

```bash
# 设置过期时间（秒）
SET session "data"
EXPIRE session 3600
# 输出: (integer) 1

# 查看剩余生存时间
TTL session
# 输出: (integer) 3598（剩余秒数）
# 输出: (integer) -1（没有设置过期时间）
# 输出: (integer) -2（键不存在）

# 设置过期时间（毫秒）
PEXPIRE session 3600000

# 移除过期时间
PERSIST session
# 输出: (integer) 1

# 设置绝对过期时间戳（Unix时间戳）
EXPIREAT session 1700000000
```

### INCR与DECR

```bash
# 递增
SET counter 10
INCR counter
# 输出: (integer) 11

# 指定步长递增
INCRBY counter 5
# 输出: (integer) 16

# 递减
DECR counter
# 输出: (integer) 15

# 指定步长递减
DECRBY counter 3
# 输出: (integer) 12

# 浮点数递增
SET price "10.5"
INCRBYFLOAT price 0.5
# 输出: "11"
```

### 其他常用命令

```bash
# 获取字符串长度
SET greeting "Hello World"
STRLEN greeting
# 输出: (integer) 11

# 追加字符串
APPEND greeting "!"
# 输出: (integer) 12
GET greeting
# 输出: "Hello World!"

# 获取子串
GETRANGE greeting 0 4
# 输出: "Hello"

# 设置子串
SETRANGE greeting 6 "Redis"
GET greeting
# 输出: "Hello Redis!"
```

## 三、注意事项与常见陷阱

1. **NX/XX参数**：实现分布式锁的关键，NX保证原子性
2. **EX/PX参数**：秒级vs毫秒级过期时间
3. **INCR是原子操作**：多客户端并发安全
4. **GET不存在的键返回nil**：不是空字符串
5. **MSET是原子的**：要么全成功要么全失败
6. **过期时间精度**：Redis过期时间精度在0-1毫秒之间
