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
