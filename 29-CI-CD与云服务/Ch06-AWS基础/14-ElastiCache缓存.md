# ElastiCache缓存

## 一、概念说明

ElastiCache是AWS的托管缓存服务，支持Redis和Memcached两种引擎。用于减轻数据库负载、提升应用性能。

| 引擎 | 特点 | 适用场景 |
|------|------|----------|
| Redis | 持久化、数据结构丰富、集群 | 会话存储、排行榜、消息队列 |
| Memcached | 多线程、简单KV、无持久化 | 纯缓存场景 |

## 二、具体用法

### 创建Redis集群

```bash
# 创建Redis复制组
aws elasticache create-replication-group \
    --replication-group-id my-redis \
    --replication-group-description "生产环境Redis" \
    --engine redis \
    --engine-version 7.0 \
    --cache-node-type cache.t3.micro \
    --num-cache-clusters 3 \
    --automatic-failover-enabled \
    --multi-az-enabled \
    --at-rest-encryption-enabled \
    --transit-encryption-enabled \
    --cache-subnet-group-name my-subnet-group \
    --security-group-ids sg-12345678
```

### 创建Memcached集群

```bash
aws elasticache create-cache-cluster \
    --cache-cluster-id my-memcached \
    --engine memcached \
    --cache-node-type cache.t3.micro \
    --num-cache-nodes 2 \
    --cache-subnet-group-name my-subnet-group
```

### 应用端连接

```python
import redis

# 连接ElastiCache Redis
r = redis.Redis(
    host='my-redis.xxxx.cache.amazonaws.com',
    port=6379,
    ssl=True,
    decode_responses=True
)

# 基本操作
r.set('key', 'value', ex=3600)
value = r.get('key')

# 缓存模式
def get_user(user_id):
    cache_key = f"user:{user_id}"
    cached = r.get(cache_key)
    if cached:
        return json.loads(cached)

    # 从数据库查询
    user = db.query(f"SELECT * FROM users WHERE id = {user_id}")
    r.setex(cache_key, 300, json.dumps(user))
    return user
```

```java
// Java Jedis连接
JedisPool pool = new JedisPool(
    new JedisPoolConfig(),
    "my-redis.xxxx.cache.amazonaws.com",
    6379,
    2000,
    null,
    true  // SSL
);

try (Jedis jedis = pool.getResource()) {
    jedis.set("key", "value");
    String value = jedis.get("key");
}
```

### 监控指标

```bash
# 查看缓存指标
aws cloudwatch get-metric-statistics \
    --namespace AWS/ElastiCache \
    --metric-name CacheHitRate \
    --dimensions Name=CacheClusterId,Value=my-redis \
    --start-time 2024-01-01T00:00:00Z \
    --end-time 2024-01-02T00:00:00Z \
    --period 3600 \
    --statistics Average
```

## 三、注意事项与常见陷阱

1. **引擎选择**：需要持久化和丰富数据结构选Redis，纯缓存选Memcached
2. **安全组配置**：仅允许应用服务器访问缓存端口
3. **内存管理**：配置淘汰策略（allkeys-lru/volatile-lru）
4. **持久化配置**：Redis启用AOF/RDB防止数据丢失
5. **连接池管理**：应用端使用连接池避免频繁建立连接
6. **多可用区**：生产环境启用多AZ和自动故障切换
7. **版本升级**：计划维护窗口进行引擎版本升级
