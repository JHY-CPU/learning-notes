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
4. **超时设置**：避免长时间阻塞
