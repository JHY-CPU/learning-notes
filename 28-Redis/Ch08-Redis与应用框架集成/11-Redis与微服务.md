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
4. **事件顺序**：Stream保证有序
