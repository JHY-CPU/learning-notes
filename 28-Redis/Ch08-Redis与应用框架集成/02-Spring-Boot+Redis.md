# Spring Boot + Redis

## 一、概念说明

Spring Boot通过spring-boot-starter-data-redis提供Redis集成，默认使用Lettuce客户端。

## 二、依赖配置

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-data-redis</artifactId>
</dependency>
```

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

## 三、基本使用

```java
@Autowired
private StringRedisTemplate redisTemplate;

// 基本操作
redisTemplate.opsForValue().set("key", "value");
String value = redisTemplate.opsForValue().get("key");

// Hash操作
redisTemplate.opsForHash().put("user:1", "name", "张三");
redisTemplate.opsForHash().get("user:1", "name");

// List操作
redisTemplate.opsForList().leftPush("queue", "task1");
redisTemplate.opsForList().rightPop("queue");

// Set操作
redisTemplate.opsForSet().add("tags", "java", "redis");
redisTemplate.opsForSet().members("tags");

// ZSet操作
redisTemplate.opsForZSet().add("rank", "player1", 100);
redisTemplate.opsForZSet().reverseRange("rank", 0, 9);
```

## 四、缓存注解

```java
@Cacheable(value = "users", key = "#id")
public User getUser(Long id) {
    return userRepository.findById(id);
}

@CachePut(value = "users", key = "#user.id")
public User updateUser(User user) {
    return userRepository.save(user);
}

@CacheEvict(value = "users", key = "#id")
public void deleteUser(Long id) {
    userRepository.deleteById(id);
}
```

## 五、注意事项

1. **序列化**：默认JDK序列化不友好，推荐JSON
2. **连接池**：合理配置连接池大小
3. **缓存穿透**：设置空值缓存
## 六、Redis序列化配置

```java
@Configuration
public class RedisConfig {
    
    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory factory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(factory);
        
        // Key序列化
        StringRedisSerializer stringSerializer = new StringRedisSerializer();
        template.setKeySerializer(stringSerializer);
        template.setHashKeySerializer(stringSerializer);
        
        // Value序列化（JSON）
        Jackson2JsonRedisSerializer<Object> jsonSerializer = 
            new Jackson2JsonRedisSerializer<>(Object.class);
        ObjectMapper mapper = new ObjectMapper();
        mapper.activateDefaultTyping(
            mapper.getPolymorphicTypeValidator(),
            ObjectMapper.DefaultTyping.NON_FINAL
        );
        jsonSerializer.setObjectMapper(mapper);
        
        template.setValueSerializer(jsonSerializer);
        template.setHashValueSerializer(jsonSerializer);
        
        template.afterPropertiesSet();
        return template;
    }
}
```

## 七、分布式锁使用

```java
@Service
public class OrderService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    public void processOrder(Long orderId) {
        String lockKey = "lock:order:" + orderId;
        String lockValue = UUID.randomUUID().toString();
        
        try {
            // 获取锁
            Boolean acquired = redisTemplate.opsForValue()
                .setIfAbsent(lockKey, lockValue, 30, TimeUnit.SECONDS);
            
            if (Boolean.TRUE.equals(acquired)) {
                // 临界区
                doProcessOrder(orderId);
            }
        } finally {
            // 释放锁（Lua脚本保证原子性）
            String lua = "if redis.call('get', KEYS[1]) == ARGV[1] then " +
                        "return redis.call('del', KEYS[1]) else return 0 end";
            redisTemplate.execute(new DefaultRedisScript<>(lua, Long.class),
                Collections.singletonList(lockKey), lockValue);
        }
    }
}
```

## 八、缓存穿透防护

```java
@Service
public class UserService {
    
    @Autowired
    private RedisTemplate<String, Object> redisTemplate;
    
    @Autowired
    private UserRepository userRepository;
    
    public User getUser(Long id) {
        String key = "user:" + id;
        
        // 1. 查缓存
        User user = (User) redisTemplate.opsForValue().get(key);
        if (user != null) {
            if (user.getName() == null) {
                return null; // 空值缓存
            }
            return user;
        }
        
        // 2. 查数据库
        user = userRepository.findById(id).orElse(null);
        
        if (user != null) {
            redisTemplate.opsForValue().set(key, user, 30, TimeUnit.MINUTES);
        } else {
            // 缓存空值
            redisTemplate.opsForValue().set(key, new User(), 5, TimeUnit.MINUTES);
        }
        
        return user;
    }
}
```
