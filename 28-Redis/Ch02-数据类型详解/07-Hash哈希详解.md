# Hash哈希详解

## 一、概念说明

Redis的Hash是一个String类型的字段和值的映射表，类似于Java中的HashMap。适合存储对象，可以独立地获取和修改单个字段，避免了序列化整个对象的开销。

## 二、具体用法

### 基本操作

```bash
# 设置单个字段
HSET user:1001 name "张三"
HSET user:1001 age 25
HSET user:1001 email "zhangsan@example.com"
# 输出: (integer) 1（新增字段）
# 输出: (integer) 0（字段已存在，更新值）

# 获取单个字段
HGET user:1001 name
# 输出: "张三"

# 获取多个字段
HMGET user:1001 name age email
# 输出: 1) "张三" 2) "25" 3) "zhangsan@example.com"

# 获取所有字段和值
HGETALL user:1001
# 输出: 1) "name" 2) "张三"
#       3) "age" 4) "25"
#       5) "email" 6) "zhangsan@example.com"
```

### 批量操作

```bash
# 批量设置字段
HSET user:1002 name "李四" age 30 email "lisi@example.com"
# 输出: (integer) 3（新增3个字段）

# 仅当字段不存在时设置
HSETNX user:1001 phone "13800138000"
# 输出: (integer) 1（成功）
HSETNX user:1001 phone "13900139000"
# 输出: (integer) 0（已存在，失败）
```

### 数值操作

```bash
# 字段值递增
HSET product:1001 stock 100
HINCRBY product:1001 stock -1
# 输出: (integer) 99

HINCRBY product:1001 stock 10
# 输出: (integer) 109

# 浮点数递增
HSET product:1001 price "99.9"
HINCRBYFLOAT product:1001 price 0.1
# 输出: "100"
```

### 删除与检查

```bash
# 删除字段
HDEL user:1001 email
# 输出: (integer) 1

# 检查字段是否存在
HEXISTS user:1001 name
# 输出: (integer) 1

HEXISTS user:1001 email
# 输出: (integer) 0

# 获取所有字段名
HKEYS user:1001
# 输出: 1) "name" 2) "age"

# 获取所有字段值
HVALS user:1001
# 输出: 1) "张三" 2) "25"

# 获取字段数量
HLEN user:1001
# 输出: (integer) 2
```

### 遍历字段

```bash
# HSCAN安全遍历
HSCAN user:1001 0 MATCH name* COUNT 10
# 输出: 1) "0" 2) 1) "name" 2) "张三"
```

## 三、底层编码

```bash
# 两种编码方式
# 1. ziplist（压缩列表）- 字段少且值小时
# 2. hashtable（哈希表）- 字段多或值大时

# 阈值配置
# hash-max-ziplist-entries 128
# hash-max-ziplist-value 64

# 查看编码
OBJECT ENCODING user:1001
# 输出: "ziplist" 或 "hashtable"
```

## 四、注意事项与常见陷阱

1. **HGETALL风险**：大Hash使用HGETALL会阻塞，使用HSCAN遍历
2. **字段数量控制**：单个Hash建议不超过500个字段
3. **HMSET已废弃**：Redis 4.0+推荐使用HSET代替
4. **HSETNX原子性**：仅当字段不存在时才设置
5. **数值操作限制**：HINCRBY只支持整数，HINCRBYFLOAT支持浮点数
