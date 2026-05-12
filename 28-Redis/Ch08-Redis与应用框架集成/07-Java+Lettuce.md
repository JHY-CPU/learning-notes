# Java + Lettuce

## 一、概念说明

Lettuce是Spring Boot默认的Redis客户端，支持响应式编程和集群模式。

## 二、基本使用

```java
// 依赖
<dependency>
    <groupId>io.lettuce</groupId>
    <artifactId>lettuce-core</artifactId>
</dependency>

// 连接
RedisClient client = RedisClient.create("redis://password@192.168.1.100:6379");
StatefulRedisConnection<String, String> connection = client.connect();
RedisCommands<String, String> commands = connection.sync();

// 基本操作
commands.set("key", "value");
String value = commands.get("key");
commands.del("key");
```

## 三、响应式连接

```java
// 响应式API
RedisClient client = RedisClient.create("redis://localhost");
StatefulRedisConnection<String, String> connection = client.connect();
RedisStringReactiveCommands<String, String> reactive = connection.reactive();

reactive.set("key", "value")
    .subscribe(System.out::println);

reactive.get("key")
    .subscribe(val -> System.out.println("Value: " + val));
```

## 四、集群支持

```java
// 集群连接
RedisClusterClient clusterClient = RedisClusterClient.create(
    RedisURI.builder()
        .withHost("192.168.1.100")
        .withPort(7000)
        .build()
);

StatefulRedisClusterConnection<String, String> conn = clusterClient.connect();
RedisAdvancedClusterCommands<String, String> commands = conn.sync();
```

## 五、Spring Boot配置

```yaml
spring:
  redis:
    lettuce:
      pool:
        max-active: 20
        max-idle: 10
        min-idle: 5
    timeout: 3000ms
```

## 六、注意事项

1. **线程安全**：Lettuce连接是线程安全的
2. **连接池**：Lettuce使用Netty连接池
3. **响应式**：支持Mono/Flux响应式编程
## 七、Pub/Sub使用

```java
import io.lettuce.core.pubsub.RedisPubSubListener;
import io.lettuce.core.pubsub.api.async.RedisPubSubAsyncCommands;

public class PubSubExample {
    
    public static void main(String[] args) {
        RedisClient client = RedisClient.create("redis://localhost");
        
        // 订阅端
        StatefulRedisPubSubConnection<String, String> subscriber = client.connectPubSub();
        subscriber.addListener(new RedisPubSubListener<String, String>() {
            public void message(String channel, String message) {
                System.out.println("频道 " + channel + ": " + message);
            }
            public void message(String pattern, String channel, String message) {
                System.out.println("模式 " + pattern + ", 频道 " + channel + ": " + message);
            }
            public void subscribed(String channel, long count) {}
            public void unsubscribed(String channel, long count) {}
            public void psubscribed(String pattern, long count) {}
            public void punsubscribed(String pattern, long count) {}
        });
        
        RedisPubSubAsyncCommands<String, String> async = subscriber.async();
        async.subscribe("notifications");
        async.psubscribe("user:*");
        
        // 发布端
        StatefulRedisConnection<String, String> publisher = client.connect();
        publisher.sync().publish("notifications", "系统维护通知");
    }
}
```

## 八、Lua脚本使用

```java
import io.lettuce.core.ScriptOutputType;

public class LuaExample {
    
    public static void main(String[] args) {
        RedisClient client = RedisClient.create("redis://localhost");
        StatefulRedisConnection<String, String> connection = client.connect();
        RedisCommands<String, String> commands = connection.sync();
        
        // Lua脚本：原子性递增并检查
        String lua = 
            "local current = redis.call('INCR', KEYS[1]) " +
            "if current == 1 then " +
            "  redis.call('EXPIRE', KEYS[1], ARGV[1]) " +
            "end " +
            "return current";
        
        Long result = commands.eval(lua, ScriptOutputType.INTEGER, 
            new String[]{"rate:user:1001"}, "60");
        
        System.out.println("当前计数: " + result);
        
        // EVALSHA方式（推荐）
        String sha = commands.scriptLoad(lua);
        result = commands.evalsha(sha, ScriptOutputType.INTEGER,
            new String[]{"rate:user:1002"}, "60");
    }
}
```

## 九、Spring Boot配置详解

```yaml
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
      shutdown-timeout: 2000ms
    
    # 哨兵配置
    sentinel:
      master: mymaster
      nodes:
        - 192.168.1.100:26379
        - 192.168.1.101:26379
        - 192.168.1.102:26379
    
    # 集群配置
    cluster:
      nodes:
        - 192.168.1.100:7000
        - 192.168.1.101:7001
        - 192.168.1.102:7002
      max-redirects: 3
```
