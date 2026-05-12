# Redis集成概述

## 一、概念说明

各编程语言都有成熟的Redis客户端库，提供了连接池、序列化、集群支持等功能。

## 二、各语言客户端对比

| 语言 | 客户端 | 特点 |
|------|--------|------|
| Java | Jedis | 简单易用，同步 |
| Java | Lettuce | 响应式，支持集群 |
| Python | redis-py | 官方推荐 |
| Node.js | ioredis | 功能丰富，支持集群 |
| Go | go-redis | 高性能，支持集群 |
| C# | StackExchange.Redis | .NET首选 |

## 三、选择建议

```bash
# Java项目
# Spring Boot → Lettuce（默认）
# 简单项目 → Jedis

# Python项目
# redis-py → 同步
# aioredis → 异步

# Node.js项目
# ioredis → 推荐
# node-redis → 官方

# Go项目
# go-redis → 推荐
# redigo → 老牌
```

## 四、通用功能

```bash
# 1. 连接池管理
# 2. 序列化/反序列化
# 3. 集群支持
# 4. 哨兵支持
# 5. 重试机制
# 6. 管道支持
# 7. 发布订阅
# 8. Lua脚本支持
```

## 五、注意事项

1. **版本兼容**：客户端版本与Redis版本匹配
2. **连接池配置**：根据并发量调整
3. **序列化选择**：JSON/Protobuf/MessagePack
## 六、客户端选型指南

```bash
# Java选型
# 场景1: Spring Boot项目 → Lettuce (默认)
# 场景2: 简单项目 → Jedis
# 场景3: 需要响应式 → Lettuce
# 场景4: 需要Redisson高级功能 → Redisson

# Python选型
# 场景1: 同步项目 → redis-py (官方)
# 场景2: 异步项目 → redis-py (asyncio) 或 aioredis
# 场景3: Django → django-redis
# 场景4: Flask → Flask-Caching

# Node.js选型
# 场景1: 通用项目 → ioredis (推荐)
# 场景2: 官方支持 → node-redis
# 场景3: 集群模式 → ioredis

# Go选型
# 场景1: 通用项目 → go-redis (推荐)
# 场景2: 老项目 → redigo
# 场景3: 高性能 → go-redis

# C#/.NET选型
# 首选 → StackExchange.Redis
# 高并发 → CSRedisCore
```

## 七、客户端通用功能对比

```bash
# 连接池
# 所有主流客户端都支持
# 配置要点：maxTotal, maxIdle, minIdle

# 集群支持
# Lettuce: ✓ 自动槽位映射
# Jedis: ✓ JedisCluster
# ioredis: ✓ Redis.Cluster
# go-redis: ✓ ClusterClient

# 哨兵支持
# Lettuce: ✓ RedisSentinel
# Jedis: ✓ JedisSentinelPool
# ioredis: ✓ Redis.Sentinel
# go-redis: ✓ FailoverClient

# Pipeline支持
# 所有主流客户端都支持
# 性能提升10-100倍

# 发布订阅
# 所有主流客户端都支持
# 注意阻塞和线程安全

# Lua脚本
# 所有主流客户端都支持
# 推荐使用EVALSHA减少网络传输

# 序列化
# Java: JSON/JDK/Protobuf/MessagePack
# Python: JSON/pickle/msgpack
# Node.js: JSON/MessagePack
# Go: JSON/MessagePack/gob
```

## 八、故障处理与降级

```python
import redis
import time

class ResilientRedisClient:
    """具备弹性的Redis客户端"""
    
    def __init__(self, host='localhost', port=6379, max_retries=3):
        self.pool = redis.ConnectionPool(
            host=host, port=port,
            max_connections=20,
            socket_timeout=5,
            socket_connect_timeout=5,
            retry_on_timeout=True
        )
        self.r = redis.Redis(connection_pool=self.pool)
        self.max_retries = max_retries
        self.failure_count = 0
    
    def execute_with_retry(self, func, *args, **kwargs):
        """带重试的执行"""
        for attempt in range(self.max_retries):
            try:
                result = func(*args, **kwargs)
                self.failure_count = 0
                return result
            except redis.ConnectionError as e:
                self.failure_count += 1
                if attempt < self.max_retries - 1:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    raise
    
    def get(self, key, default=None):
        """获取值（带降级）"""
        try:
            return self.execute_with_retry(self.r.get, key)
        except Exception as e:
            print(f"Redis错误: {e}")
            return default
    
    def set(self, key, value, ex=None):
        """设置值（带降级）"""
        try:
            return self.execute_with_retry(self.r.set, key, value, ex=ex)
        except Exception as e:
            print(f"Redis错误: {e}")
            return False
```
