# Redis微服务集成

## 一、概念说明

Redis在微服务架构中充当缓存层、消息队列、分布式锁等多种角色。需要与各微服务框架集成。

## 二、Spring Boot集成

```yaml
# application.yml
spring:
  redis:
    host: 192.168.1.100
    port: 6379
    password: yourpassword
    database: 0
    timeout: 3000ms
    lettuce:
      pool:
        max-active: 20
        max-idle: 10
        min-idle: 5
        max-wait: 3000ms
```

```java
// 使用RedisTemplate
@Autowired
private RedisTemplate<String, Object> redisTemplate;

// 缓存注解
@Cacheable(value = "users", key = "#id")
public User getUser(Long id) {
    return userRepository.findById(id);
}
```

## 三、Node.js集成

```javascript
const Redis = require('ioredis');

const redis = new Redis({
    host: '192.168.1.100',
    port: 6379,
    password: 'yourpassword',
    db: 0,
    retryStrategy(times) {
        return Math.min(times * 50, 2000);
    }
});

// 使用
await redis.set('key', 'value');
const value = await redis.get('key');
```

## 四、Go集成

```go
import "github.com/go-redis/redis/v9"

client := redis.NewClient(&redis.Options{
    Addr:     "192.168.1.100:6379",
    Password: "yourpassword",
    DB:       0,
})

// 使用
client.Set(ctx, "key", "value", time.Hour)
val, _ := client.Get(ctx, "key").Result()
```

## 五、注意事项

1. **连接池**：合理配置连接池大小
2. **超时设置**：设置合理的连接和读写超时
3. **重试机制**：配置失败重试策略
4. **监控**：监控连接池使用情况
5. **多实例**：微服务各自维护连接池

## 六、微服务缓存策略

```python
import redis
import json
from functools import wraps

r = redis.Redis()

class MicroserviceCache:
    """微服务缓存管理器"""
    
    def __init__(self, service_name):
        self.service_name = service_name
        self.r = redis.Redis()
    
    def cache_key(self, resource, resource_id=None):
        """生成缓存Key"""
        if resource_id:
            return f"{self.service_name}:{resource}:{resource_id}"
        return f"{self.service_name}:{resource}"
    
    def get_or_set(self, resource, resource_id, db_query_func, ttl=3600):
        """获取或设置缓存"""
        key = self.cache_key(resource, resource_id)
        
        cached = self.r.get(key)
        if cached:
            return json.loads(cached)
        
        value = db_query_func(resource_id)
        if value:
            self.r.setex(key, ttl, json.dumps(value))
        
        return value
    
    def invalidate(self, resource, resource_id):
        """使缓存失效"""
        key = self.cache_key(resource, resource_id)
        self.r.delete(key)
    
    def invalidate_pattern(self, pattern):
        """批量使缓存失效"""
        cursor = 0
        while True:
            cursor, keys = self.r.scan(cursor, match=pattern, count=100)
            if keys:
                self.r.delete(*keys)
            if cursor == 0:
                break
    
    def cache_decorator(self, resource, ttl=3600):
        """缓存装饰器"""
        def decorator(func):
            @wraps(func)
            def wrapper(resource_id, *args, **kwargs):
                key = self.cache_key(resource, resource_id)
                cached = self.r.get(key)
                
                if cached:
                    return json.loads(cached)
                
                result = func(resource_id, *args, **kwargs)
                if result:
                    self.r.setex(key, ttl, json.dumps(result))
                
                return result
            return wrapper
        return decorator

# 使用
user_cache = MicroserviceCache("user-service")
order_cache = MicroserviceCache("order-service")

# 获取用户（带缓存）
user = user_cache.get_or_set("user", 1001, query_user)

# 缓存失效
user_cache.invalidate("user", 1001)
```

## 七、微服务事件总线

```python
class EventBus:
    """Redis事件总线"""
    
    def __init__(self):
        self.r = redis.Redis()
        self.handlers = {}
    
    def subscribe(self, event_type, handler):
        """订阅事件"""
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def publish(self, event_type, data):
        """发布事件"""
        message = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self.r.publish(f"events:{event_type}", json.dumps(message))
    
    def start_listening(self):
        """启动监听"""
        pubsub = self.r.pubsub()
        
        # 订阅所有事件
        channels = [f"events:{et}" for et in self.handlers.keys()]
        pubsub.subscribe(*channels)
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                event = json.loads(message['data'])
                event_type = event['type']
                
                if event_type in self.handlers:
                    for handler in self.handlers[event_type]:
                        try:
                            handler(event['data'])
                        except Exception as e:
                            print(f"处理事件失败: {e}")

# 使用
bus = EventBus()

def on_user_created(data):
    print(f"用户创建事件: {data}")

bus.subscribe('user.created', on_user_created)
bus.publish('user.created', {'user_id': 1001, 'name': '张三'})
```

## 八、分布式配置中心

```python
class DistributedConfig:
    """分布式配置管理"""
    
    def __init__(self, service_name):
        self.service_name = service_name
        self.r = redis.Redis()
        self.config_key = f"config:{service_name}"
    
    def get(self, key, default=None):
        """获取配置"""
        value = self.r.hget(self.config_key, key)
        if value:
            return json.loads(value)
        return default
    
    def set(self, key, value):
        """设置配置"""
        self.r.hset(self.config_key, key, json.dumps(value))
        # 通知配置变更
        self.r.publish(f"config:changed:{self.service_name}", key)
    
    def get_all(self):
        """获取所有配置"""
        data = self.r.hgetall(self.config_key)
        return {k.decode(): json.loads(v) for k, v in data.items()}
    
    def watch(self, callback):
        """监听配置变更"""
        pubsub = self.r.pubsub()
        pubsub.subscribe(f"config:changed:{self.service_name}")
        
        for message in pubsub.listen():
            if message['type'] == 'message':
                key = message['data'].decode()
                value = self.get(key)
                callback(key, value)

# 使用
config = DistributedConfig("user-service")
config.set("max_connections", 100)
config.set("timeout", 30)

# 监听变更
def on_config_change(key, value):
    print(f"配置变更: {key} = {value}")

config.watch(on_config_change)
```
