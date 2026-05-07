# Redis测试策略

## 一、概念说明

Redis测试包括单元测试、集成测试等。可以使用嵌入式Redis或Mock来测试Redis相关代码。

## 二、嵌入式Redis

```java
// 依赖
<dependency>
    <groupId>it.ozimov</groupId>
    <artifactId>embedded-redis</artifactId>
    <version>0.7.3</version>
</dependency>

// 测试配置
@TestConfiguration
public class TestRedisConfig {
    private RedisServer redisServer;
    
    @PostConstruct
    public void startRedis() {
        redisServer = new RedisServer(6379);
        redisServer.start();
    }
    
    @PreDestroy
    public void stopRedis() {
        redisServer.stop();
    }
}
```

## 三、Testcontainers

```java
@SpringBootTest
@Testcontainers
public class RedisIntegrationTest {
    @Container
    static GenericContainer<?> redis = 
        new GenericContainer<>("redis:7.2-alpine")
            .withExposedPorts(6379);
    
    @Test
    public void testRedis() {
        RedisClient client = RedisClient.create(
            "redis://" + redis.getHost() + ":" + redis.getFirstMappedPort()
        );
        // 测试...
    }
}
```

## 四、Mock测试

```java
@MockBean
private RedisTemplate<String, Object> redisTemplate;

@Test
public void testWithMockRedis() {
    when(redisTemplate.opsForValue().get("key"))
        .thenReturn("value");
    
    // 测试业务逻辑
}
```

## 五、Python测试

```python
import fakeredis

def test_redis_operations():
    r = fakeredis.FakeRedis()
    
    r.set('key', 'value')
    assert r.get('key') == b'value'
    
    r.hset('hash', 'field', 'value')
    assert r.hget('hash', 'field') == b'value'
```

## 六、注意事项

1. **隔离性**：测试环境与生产环境隔离
2. **清理数据**：测试后清理Redis数据
3. **并发测试**：测试并发场景
4. **性能测试**：使用redis-benchmark
