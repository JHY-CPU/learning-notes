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
## 七、集成测试完整示例

```java
@SpringBootTest
@Testcontainers
public class RedisIntegrationTest {
    
    @Container
    static GenericContainer<?> redis = new GenericContainer<>("redis:7.2-alpine")
        .withExposedPorts(6379);
    
    @DynamicPropertySource
    static void configureRedis(DynamicPropertyRegistry registry) {
        registry.add("spring.redis.host", redis::getHost);
        registry.add("spring.redis.port", redis::getFirstMappedPort);
    }
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    @Test
    public void testBasicOperations() {
        // String操作
        redisTemplate.opsForValue().set("key", "value");
        assertEquals("value", redisTemplate.opsForValue().get("key"));
        
        // Hash操作
        redisTemplate.opsForHash().put("user", "name", "张三");
        assertEquals("张三", redisTemplate.opsForHash().get("user", "name"));
        
        // 过期时间
        redisTemplate.expire("key", 60, TimeUnit.SECONDS);
        assertTrue(redisTemplate.getExpire("key") > 0);
    }
    
    @Test
    public void testCacheAside() {
        // 模拟缓存穿透
        String key = "user:9999";
        Object value = redisTemplate.opsForValue().get(key);
        assertNull(value);
        
        // 缓存空值
        redisTemplate.opsForValue().set(key, "NULL", 5, TimeUnit.MINUTES);
        assertEquals("NULL", redisTemplate.opsForValue().get(key));
    }
}
```

## 八、负载测试

```python
import redis
import time
import threading
from concurrent.futures import ThreadPoolExecutor

r = redis.Redis()

def load_test(thread_count=10, operations=10000):
    """负载测试"""
    ops_per_thread = operations // thread_count
    results = []
    
    def worker():
        local_r = redis.Redis()
        start = time.time()
        for i in range(ops_per_thread):
            local_r.set(f"key:{i}", f"value:{i}")
            local_r.get(f"key:{i}")
        elapsed = time.time() - start
        results.append(ops_per_thread / elapsed)
    
    with ThreadPoolExecutor(max_workers=thread_count) as executor:
        futures = [executor.submit(worker) for _ in range(thread_count)]
        for future in futures:
            future.result()
    
    total_qps = sum(results)
    print(f"线程数: {thread_count}")
    print(f"总QPS: {total_qps:.0f}")
    print(f"平均延迟: {1000/total_qps*thread_count:.2f}ms")

# 执行测试
load_test(thread_count=10, operations=10000)
load_test(thread_count=50, operations=50000)
load_test(thread_count=100, operations=100000)
```

## 九、测试最佳实践

```bash
# 1. 测试环境隔离
# 使用独立的Redis实例
# 测试前后清理数据

# 2. 使用Testcontainers
# 自动管理Redis容器
# 每次测试获得干净环境

# 3. Mock测试
# 使用fakeredis进行单元测试
# 不依赖真实Redis

# 4. 集成测试
# 测试缓存逻辑
# 测试分布式锁
# 测试消息队列

# 5. 性能测试
# 使用redis-benchmark
# 测试不同并发级别
# 记录基线性能数据

# 6. 故障测试
# 模拟Redis不可用
# 测试降级逻辑
# 测试重试机制
```
