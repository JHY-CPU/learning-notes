# 消息 TTL

## 一、消息过期时间

TTL（Time To Live）定义消息在队列中的最大存活时间。

```java
// RabbitMQ 队列级别 TTL
@Bean
public Queue ttlQueue() {
    return QueueBuilder.durable("ttl-queue")
        .withArgument("x-message-ttl", 60000)  // 60 秒
        .build();
}

// RabbitMQ 消息级别 TTL
rabbitTemplate.convertAndSend("exchange", "key", msg, m -> {
    m.getMessageProperties().setExpiration("30000");  // 30 秒
    return m;
});

// RocketMQ 延迟级别
message.setDelayTimeLevel(5);  // 1 分钟后过期
```

## 二、队列保留策略

```properties
# Kafka Topic 保留策略
retention.ms=604800000        # 保留 7 天
retention.bytes=107374182400  # 保留 100GB
cleanup.policy=delete         # 过期删除
# cleanup.policy=compact      # 按 Key 保留最新
```

## 三、注意事项

1. **RabbitMQ 的 TTL 过期消息不会立即删除**，只在被消费时检查
2. **队头消息 TTL 会阻塞后续消息**，先入队的长 TTL 消息阻塞短 TTL 消息
3. **Kafka 保留策略是 Topic 级别的**，不支持消息级别
4. **日志压缩（compact）保留每个 Key 的最新值**
5. **TTL 结合死信队列使用效果更好**
