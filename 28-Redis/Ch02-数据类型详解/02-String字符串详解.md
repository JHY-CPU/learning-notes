# String字符串详解

## 一、概念说明

String是Redis最基本的数据类型，可以存储字符串、整数或浮点数。最大容量为512MB。底层使用SDS（Simple Dynamic String）实现，支持丰富的操作命令。

## 二、具体用法

### 基本操作

```bash
# 设置与获取
SET name "Redis"
GET name
# 输出: "Redis"

# 设置带过期时间
SET session "token123" EX 3600

# 仅当key不存在时设置
SET lock:resource "locked" NX EX 10

# 仅当key存在时修改
SET name "NewRedis" XX

# 删除
DEL name
```

### 批量操作

```bash
# 批量设置
MSET key1 "val1" key2 "val2" key3 "val3"
# 输出: OK

# 批量获取
MGET key1 key2 key3
# 输出: 1) "val1" 2) "val2" 3) "val3"

# 批量设置（仅不存在时）
MSETNX key1 "newval1" key4 "val4"
# 输出: (integer) 0（key1已存在，全部失败）
```

### 数值操作

```bash
# 整数递增
SET counter 10
INCR counter
# 输出: (integer) 11

INCRBY counter 5
# 输出: (integer) 16

# 整数递减
DECR counter
# 输出: (integer) 15

DECRBY counter 3
# 输出: (integer) 12

# 浮点数操作
SET price "10.5"
INCRBYFLOAT price 0.5
# 输出: "11"

INCRBYFLOAT price -1.5
# 输出: "9.5"
```

### 字符串操作

```bash
# 字符串长度
SET greeting "Hello World"
STRLEN greeting
# 输出: (integer) 11

# 追加字符串
APPEND greeting "!"
# 输出: (integer) 12

# 获取子串
GETRANGE greeting 0 4
# 输出: "Hello"

# 设置子串
SETRANGE greeting 6 "Redis"
GET greeting
# 输出: "Hello Redis!"
```

## 三、底层编码

```bash
# 三种编码方式
# 1. int - 纯整数值（8字节长整型范围内）
# 2. embstr - 短字符串（<=44字节）
# 3. raw - 长字符串（>44字节）

# 查看编码
SET num 12345
OBJECT ENCODING num
# 输出: "int"

SET short "hello"
OBJECT ENCODING short
# 输出: "embstr"

SET long "a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]a]"
OBJECT ENCODING long
# 输出: "raw"
```

## 四、应用场景

```bash
# 1. 缓存
SET user:1001 '{"name":"张三"}' EX 3600

# 2. 计数器
INCR article:1001:views

# 3. 分布式锁
SET lock:order:1001 "uuid" NX EX 30

# 4. 限流器
INCR rate:limit:user1001
EXPIRE rate:limit:user1001 60
```

## 五、注意事项与常见陷阱

1. **INCR是原子操作**：多线程安全，无需额外同步
2. **字符串追加是O(N)**：频繁追加大字符串有性能问题
3. **GETRANGE支持负数索引**：-1表示最后一个字符
4. **MSET原子性**：要么全成功要么全失败
5. **embstr是只读的**：修改时会转换为raw编码
