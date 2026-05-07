# Spring RedisTemplate

## 一、概念说明

RedisTemplate是Spring Data Redis的核心类，提供了丰富的Redis操作API。

## 二、配置序列化

```java
@Configuration
public class RedisConfig {
    @Bean
    public RedisTemplate<String, Object> redisTemplate(
            RedisConnectionFactory factory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(factory);
        
        // JSON序列化
        Jackson2JsonRedisSerializer<Object> serializer = 
            new Jackson2JsonRedisSerializer<>(Object.class);
        
        // Key序列化
        template.setKeySerializer(new StringRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        
        // Value序列化
        template.setValueSerializer(serializer);
        template.setHashValueSerializer(serializer);
        
        return template;
    }
}
```

## 三、操作模板

```java
// ValueOperations
ValueOperations<String, Object> ops = redisTemplate.opsForValue();
ops.set("key", "value", 30, TimeUnit.MINUTES);
ops.increment("counter", 1);

// HashOperations
HashOperations<String, Object, Object> hashOps = redisTemplate.opsForHash();
hashOps.putAll("user:1", Map.of("name", "张三", "age", 25));

// ListOperations
ListOperations<String, Object> listOps = redisTemplate.opsForList();
listOps.leftPush("queue", "task");
listOps.range("queue", 0, -1);

// SetOperations
SetOperations<String, Object> setOps = redisTemplate.opsForSet();
setOps.add("tags", "java", "redis");

// ZSetOperations
ZSetOperations<String, Object> zSetOps = redisTemplate.opsForZSet();
zSetOps.add("rank", "player1", 100);
zSetOps.reverseRange("rank", 0, 9);
```

## 四、Pipeline使用

```java
List<Object> results = redisTemplate.executePipelined(
    (RedisCallback<Object>) connection -> {
        StringRedisConnection stringConn = (StringRedisConnection) connection;
        for (int i = 0; i < 1000; i++) {
            stringConn.set("key:" + i, "value:" + i);
        }
        return null;
    }
);
```

## 五、注意事项

1. **序列化一致**：读写使用相同序列化方式
2. **类型安全**：使用泛型保证类型安全
3. **事务支持**：使用@Transactional
4. **连接池**：Lettuce默认支持连接池
