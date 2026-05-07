# Hash应用场景

## 一、概念说明

Hash类型适合存储对象数据，可以独立操作每个字段，避免了整体序列化和反序列化的开销。在存储用户信息、商品信息等场景下非常高效。

## 二、核心应用场景

### 用户信息存储

```bash
# 存储用户信息
HSET user:1001 name "张三"
HSET user:1001 age 25
HSET user:1001 email "zhangsan@example.com"
HSET user:1001 avatar "/images/avatar/1001.jpg"
HSET user:1001 created_at "2024-01-01"

# 获取单个字段（高效）
HGET user:1001 name
# 输出: "张三"

# 获取所有字段
HGETALL user:1001

# 更新单个字段
HSET user:1001 age 26

# 删除字段
HDEL user:1001 email
```

### 对象属性存储

```bash
# 商品信息
HSET product:1001 name "iPhone 15"
HSET product:1001 price "7999"
HSET product:1001 stock 500
HSET product:1001 category "手机"
HSET product:1001 brand "Apple"

# 库存扣减
HINCRBY product:1001 stock -1
# 输出: (integer) 499

# 价格更新
HSET product:1001 price "7499"
```

### 配置信息存储

```bash
# 系统配置
HSET config:app version "1.0.0"
HSET config:app maintenance "false"
HSET config:app max_users 10000
HSET config:app session_timeout 1800

# 获取配置
HGET config:app maintenance
# 输出: "false"

# 动态修改配置
HSET config:app maintenance "true"
```

### 统计计数器

```bash
# 用户行为统计
HINCRBY stats:user:1001 login_count 1
HINCRBY stats:user:1001 post_count 1
HINCRBY stats:user:1001 like_count 5

# 获取统计结果
HMGET stats:user:1001 login_count post_count like_count
# 输出: 1) "10" 2) "25" 3) "150"

# 文章统计
HINCRBY stats:article:1001 views 1
HINCRBY stats:article:1001 likes 1
```

### 对比String存储对象

```bash
# String方式（整体序列化）
SET user:1001 '{"name":"张三","age":25,"email":"zhangsan@example.com"}'
# 修改年龄需要反序列化->修改->序列化->存储

# Hash方式（独立字段）
HSET user:1001 name "张三"
HSET user:1001 age 25
HSET user:1001 email "zhangsan@example.com"
# 修改年龄直接 HSET user:1001 age 26

# Hash优势：
# 1. 可以单独修改字段
# 2. 可以单独获取字段
# 3. 节省内存（ziplist编码）
# 4. 支持字段级别的原子操作
```

## 三、最佳实践

```bash
# 1. 字段数量控制
# 建议不超过500个字段
# 超过则考虑分片

# 2. 合理的Key设计
# user:{id}:profile
# product:{id}:info
# config:{module}:{key}

# 3. 设置过期时间
# Hash整体设置过期，不能单字段设置
EXPIRE user:1001 3600

# 4. 避免HGETALL大Hash
# 使用HSCAN遍历大Hash
HSCAN user:1001 0 COUNT 100
```

## 四、注意事项与常见陷阱

1. **不能单字段设置过期**：过期时间作用于整个Hash
2. **HGETALL性能**：大Hash使用HGETALL会阻塞
3. **内存优化**：小Hash使用ziplist编码更省内存
4. **字段命名**：保持字段名简短一致
5. **数据一致性**：修改多个字段不是原子操作
