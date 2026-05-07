# String应用场景

## 一、概念说明

String类型是Redis最灵活的数据类型，广泛应用于缓存、计数器、分布式锁等场景。掌握String的应用模式是使用Redis的基础。

## 二、核心应用场景

### 缓存数据

```bash
# 缓存数据库查询结果
# 序列化对象后存储
SET user:1001 '{"id":1001,"name":"张三","age":25}' EX 3600

# 获取缓存
GET user:1001
# 输出: "{\"id\":1001,\"name\":\"张三\",\"age\":25}"

# 缓存穿透防护：存储空值
GET user:9999
# 输出: (nil)
# 查数据库 -> 不存在 -> SET user:9999 "NULL" EX 300
```

### 计数器

```bash
# 文章阅读量
INCR article:1001:views
# 输出: (integer) 1

# 获取当前值
GET article:1001:views
# 输出: "1"

# 商品库存扣减
SET stock:product:1001 100
DECR stock:product:1001
# 输出: (integer) 99

# 限流计数器
INCR rate:limit:user:1001
EXPIRE rate:limit:user:1001 60
# 每分钟限制100次
GET rate:limit:user:1001
# 如果 > 100 则拒绝
```

### 分布式锁

```bash
# 获取锁（原子操作）
SET lock:order:1001 "uuid-value" NX EX 30
# 输出: OK（获取成功）
# 输出: (nil)（锁已被占用）

# 释放锁（Lua脚本保证原子性）
EVAL "if redis.call('get',KEYS[1]) == ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end" 1 lock:order:1001 "uuid-value"

# 设置合理的过期时间
# 避免锁未释放导致死锁
# 根据业务执行时间设置，建议30秒以内
```

### 分布式Session

```bash
# 存储用户Session
SET session:token:abc123 '{"userId":1001,"loginTime":1700000000}' EX 1800

# 验证Session
GET session:token:abc123

# 刷新Session过期时间
EXPIRE session:token:abc123 1800

# 登出时删除
DEL session:token:abc123
```

### 布尔标记

```bash
# 功能开关
SET feature:new_ui "true" EX 86400

# 维护模式
SET system:maintenance "true"

# 检查标记
GET feature:new_ui
# 输出: "true"

# 用户状态标记
SET user:1001:online "1" EX 300
GET user:1001:online
# 输出: "1"（在线）
```

## 三、最佳实践

```bash
# 1. 设置过期时间避免内存泄漏
SET cache:key "value" EX 3600

# 2. 使用前缀命名空间
SET app:module:type:id "value"

# 3. 控制Value大小
# 单个Value不超过10MB
STRLEN large_key

# 4. 选择合适的序列化
# JSON：可读性好
# Protobuf：体积小
# MessagePack：速度快
```

## 四、注意事项与常见陷阱

1. **INCR是原子操作**：多客户端并发安全
2. **NX/XX参数**：实现分布式锁的关键
3. **过期时间必须设置**：避免缓存数据永久占用内存
4. **Value大小控制**：超过10MB影响性能
5. **序列化一致性**：读写使用相同的序列化方式
