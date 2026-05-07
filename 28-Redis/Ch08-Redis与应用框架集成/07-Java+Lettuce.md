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
4. **集群**：自动处理槽位映射
