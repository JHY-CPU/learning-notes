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

## 三、死信队列 + TTL 实现延迟消息

```java
// RabbitMQ 延迟消息（死信队列方案）
@Configuration
public class DelayQueueConfig {
    // 死信交换机
    @Bean
    public DirectExchange delayExchange() {
        return new DirectExchange("delay-exchange");
    }

    // 1 分钟延迟队列
    @Bean
    public Queue delayQueue1m() {
        return QueueBuilder.durable("delay-1m")
            .withArgument("x-message-ttl", 60000)
            .withArgument("x-dead-letter-exchange", "real-exchange")
            .withArgument("x-dead-letter-routing-key", "real-queue")
            .build();
    }

    // 10 分钟延迟队列
    @Bean
    public Queue delayQueue10m() {
        return QueueBuilder.durable("delay-10m")
            .withArgument("x-message-ttl", 600000)
            .withArgument("x-dead-letter-exchange", "real-exchange")
            .withArgument("x-dead-letter-routing-key", "real-queue")
            .build();
    }
}

// 发送延迟消息
public void sendDelayedMessage(String message, long delayMs) {
    String queueName = selectDelayQueue(delayMs);
    rabbitTemplate.convertAndSend("delay-exchange", queueName, message);
}

private String selectDelayQueue(long delayMs) {
    if (delayMs <= 60000) return "delay-1m";
    if (delayMs <= 600000) return "delay-10m";
    return "delay-30m";
}
```

## 四、RocketMQ 延迟消息

```java
// RocketMQ 18 个延迟级别
// 1s 5s 10s 30s 1m 2m 3m 4m 5m 6m 7m 8m 9m 10m 20m 30m 1h 2h

// 发送延迟消息
public void sendDelayMessage(String topic, Object data, int delayLevel) {
    Message msg = new Message(topic, JSON.toJSONBytes(data));
    msg.setDelayTimeLevel(delayLevel);  // 级别 3 = 10 秒后投递
    producer.send(msg);
}

// RocketMQ 5.0 支持指定时间延迟
public void sendScheduledMessage(String topic, Object data, long delayMs) {
    Message msg = new Message(topic, JSON.toJSONBytes(data));
    msg.setDeliverTimeMs(System.currentTimeMillis() + delayMs);
    producer.send(msg);
}
```

## 五、Kafka 消息保留策略调优

```properties
# 时间保留 + 大小保留组合
retention.ms=604800000          # 7 天
retention.bytes=53687091200     # 50GB
segment.bytes=1073741824        # 1GB 分段
segment.ms=86400000             # 1 天分段

# 按 Key 保留最新值（适合状态更新场景）
cleanup.policy=compact
min.cleanable.dirty.ratio=0.5
delete.retention.ms=86400000
```

## 六、注意事项

1. **RabbitMQ 的 TTL 过期消息不会立即删除**，只在被消费时检查
2. **队头消息 TTL 会阻塞后续消息**，先入队的长 TTL 消息阻塞短 TTL 消息
3. **Kafka 保留策略是 Topic 级别的**，不支持消息级别
4. **日志压缩（compact）保留每个 Key 的最新值**
5. **TTL 结合死信队列使用效果更好**
6. **定期监控队列中过期消息的数量**，异常增长需排查
7. **TTL 设置要结合业务 SLA**，避免消息过早过期或长期积压
