# Redis与微服务

## 一、概念说明

Redis在微服务架构中扮演缓存、消息队列、分布式锁等多种角色。

## 二、配置中心

```bash
# 使用Redis存储配置
HSET config:service1 max_connections 100
HSET config:service1 timeout 30

# 配置变更通知
PUBLISH config:changed "service1"

# 服务订阅
SUBSCRIBE config:changed
```

## 三、服务注册发现

```python
def register_service(name, host, port):
    key = f"service:{name}:{host}:{port}"
    r.hset(key, mapping={'host': host, 'port': port, 'status': 'up'})
    r.expire(key, 30)  # 30秒过期

def discover_service(name):
    pattern = f"service:{name}:*"
    services = []
    for key in r.scan_iter(match=pattern):
        services.append(r.hgetall(key))
    return services

# 心跳续期
def heartbeat(service_key):
    r.expire(service_key, 30)
```

## 四、分布式锁

```java
// Redisson分布式锁
RLock lock = redisson.getLock("resource:lock");
try {
    lock.lock(10, TimeUnit.SECONDS);
    // 临界区
} finally {
    lock.unlock();
}
```

## 五、事件总线

```bash
# 发布事件
PUBLISH events:order "{'type':'created','order_id':1001}"

# 订阅事件
SUBSCRIBE events:order
```

## 六、注意事项

1. **服务发现**：定期心跳续期
2. **配置热更新**：订阅配置变更
3. **分布式锁**：避免死锁
## 七、微服务缓存策略

```python
import redis
import json
from functools import wraps

class MicroserviceCache:
    """微服务缓存管理"""
    
    def __init__(self, service_name, host='localhost', port=6379):
        self.service_name = service_name
        self.r = redis.Redis(host=host, port=port)
    
    def _make_key(self, resource, resource_id=None):
        if resource_id:
            return f"{self.service_name}:{resource}:{resource_id}"
        return f"{self.service_name}:{resource}"
    
    def get_or_set(self, resource, resource_id, query_func, ttl=3600):
        key = self._make_key(resource, resource_id)
        
        cached = self.r.get(key)
        if cached:
            return json.loads(cached)
        
        value = query_func(resource_id)
        if value:
            self.r.setex(key, ttl, json.dumps(value))
        
        return value
    
    def invalidate(self, resource, resource_id=None):
        if resource_id:
            key = self._make_key(resource, resource_id)
            self.r.delete(key)
        else:
            pattern = f"{self.service_name}:{resource}:*"
            for key in self.r.scan_iter(match=pattern, count=100):
                self.r.delete(key)
    
    def cache_decorator(self, resource, ttl=3600):
        def decorator(func):
            @wraps(func)
            def wrapper(resource_id, *args, **kwargs):
                return self.get_or_set(
                    resource, resource_id,
                    lambda rid: func(rid, *args, **kwargs),
                    ttl
                )
            return wrapper
        return decorator

# 使用
user_cache = MicroserviceCache("user-service")

@user_cache.cache_decorator("user", ttl=1800)
def get_user(user_id):
    return db.query_user(user_id)
```

## 八、微服务限流

```python
class MicroserviceRateLimiter:
    """微服务限流器"""
    
    def __init__(self):
        self.r = redis.Redis()
    
    def limit(self, service, method, limit=100, window=60):
        """限流检查"""
        key = f"rate:{service}:{method}"
        
        lua = """
        local key = KEYS[1]
        local limit = tonumber(ARGV[1])
        local window = tonumber(ARGV[2])
        
        local current = redis.call('INCR', key)
        if current == 1 then
            redis.call('EXPIRE', key, window)
        end
        
        if current > limit then
            return 0
        else
            return 1
        end
        """
        
        allowed = self.r.eval(lua, 1, key, limit, window)
        return allowed == 1

# 使用
limiter = MicroserviceRateLimiter()

def api_handler(service, method):
    if not limiter.limit(service, method, limit=1000, window=60):
        return {"error": "限流"}, 429
    # 处理请求
```

## 九、微服务监控集成

```python
import redis
import time

class RedisMetrics:
    """Redis指标收集"""
    
    def __init__(self):
        self.r = redis.Redis()
    
    def collect(self):
        """收集指标"""
        info = self.r.info()
        
        metrics = {
            'memory_used_bytes': info['used_memory'],
            'memory_max_bytes': info.get('maxmemory', 0),
            'connected_clients': info['connected_clients'],
            'ops_per_sec': info['instantaneous_ops_per_sec'],
            'keyspace_hits': info['keyspace_hits'],
            'keyspace_misses': info['keyspace_misses'],
            'total_connections': info['total_connections_received'],
            'rejected_connections': info['rejected_connections'],
        }
        
        # 计算命中率
        hits = metrics['keyspace_hits']
        misses = metrics['keyspace_misses']
        total = hits + misses
        metrics['hit_rate'] = hits / total if total > 0 else 0
        
        return metrics
    
    def export_prometheus(self):
        """导出Prometheus格式"""
        metrics = self.collect()
        lines = []
        for key, value in metrics.items():
            lines.append(f'redis_{key} {value}')
        return '\n'.join(lines)
```
