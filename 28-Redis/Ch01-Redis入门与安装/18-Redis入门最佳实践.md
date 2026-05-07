# Redis入门最佳实践

## 一、Key命名规范

### 命名规则

```bash
# 推荐格式：业务:类型:标识
# 使用冒号分隔命名空间

# 用户相关
user:1001:profile
user:1001:session
user:1001:favorites

# 商品相关
product:2001:detail
product:2001:stock
product:2001:views

# 系统相关
system:config:database
system:counter:login
system:rate:limit:192.168.1.1
```

### 命名原则

```bash
# 1. 简洁明了
# 不好: user_profile_information_for_user_1001
# 好: user:1001:profile

# 2. 避免过长
# 不好: very_long_descriptive_key_name_that_takes_too_much_memory
# 好: u:1001:p

# 3. 使用小写字母和冒号
# 不好: UserProfile_1001
# 好: user:1001:profile

# 4. 包含业务标识
# 不好: cache:data
# 好: order:cache:1001
```

## 二、数据设计规范

### Hash存储对象

```bash
# 存储用户信息
HSET user:1001 name "张三"
HSET user:1001 age 25
HSET user:1001 email "zhangsan@example.com"

# 获取单个字段
HGET user:1001 name
# 输出: "张三"

# 获取所有字段
HGETALL user:1001

# 设置多个字段
HMSET user:1002 name "李四" age 30 email "lisi@example.com"
```

### 设置合理的过期时间

```bash
# 短期缓存：分钟级
SET hot:news "content" EX 300    # 5分钟

# 中期缓存：小时级
SET user:profile "data" EX 3600  # 1小时

# 长期缓存：天级
SET config:system "data" EX 86400  # 1天

# 永久数据：不设置过期时间
SET permanent:data "value"
```

## 三、性能优化建议

```bash
# 1. 使用Pipeline批量操作
# 减少网络往返次数
redis-cli --pipe < commands.txt

# 2. 避免大Key
# 单个Key的Value不超过10MB
# 单个Hash/Set不超过5000个元素

# 3. 使用合适的数据类型
# 简单键值用String
# 对象用Hash
# 列表用List
# 集合用Set
# 排序用ZSet

# 4. 避免耗时命令
# 不要使用KEYS *
# 使用SCAN替代
# 不要使用HGETALL获取大Hash
# 使用HSCAN遍历
```

## 四、安全最佳实践

```bash
# 1. 设置强密码
requirepass "YourStr0ng!Pass#2024"

# 2. 绑定内网IP
bind 192.168.1.100 127.0.0.1

# 3. 重命名危险命令
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command CONFIG ""

# 4. 使用ACL限制权限
ACL SETUSER app on >password ~app:* +@read +@write

# 5. 启用SSL/TLS（跨网络时）
tls-port 6379
tls-cert-file /path/to/cert.pem
tls-key-file /path/to/key.pem
```

## 五、监控与运维

```bash
# 定期检查的指标
INFO memory | grep used_memory_human
INFO stats | grep instantaneous_ops_per_sec
INFO clients | grep connected_clients

# 慢查询日志
SLOWLOG GET 10

# 内存碎片率
INFO memory | grep mem_fragmentation_ratio
# 正常范围: 1.0 - 1.5

# 键空间信息
INFO keyspace
# 输出: db0:keys=1000,expires=500,avg_ttl=3600000
```

## 六、常见陷阱总结

1. **不要用KEYS命令**：生产环境使用SCAN
2. **不要存储大Value**：控制单个Key大小
3. **不要忘记设置过期时间**：避免内存泄漏
4. **不要在公网暴露Redis**：使用防火墙限制
5. **不要使用默认端口**：减少扫描攻击
6. **不要忽视内存监控**：设置maxmemory和淘汰策略
7. **不要在事务中执行耗时命令**：会阻塞其他客户端
