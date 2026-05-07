# RabbitMQ 最佳实践

## 一、架构设计原则

```yaml
# 队列设计原则
队列设计:
  - 一个队列只承载一种业务消息
  - 优先使用 Quorum Queue
  - 设置合理的队列长度限制
  - 配置死信队列作为兜底

交换机设计:
  - 使用 Topic Exchange 获得最大灵活性
  - Routing Key 命名规范化: {domain}.{entity}.{action}
  - 避免过多 binding，影响路由性能

绑定设计:
  - 通配符绑定比精确绑定灵活
  - 一条队列可以有多个 binding
```

## 二、生产环境配置

```yaml
# application-prod.yml
spring:
  rabbitmq:
    host: rabbitmq-cluster.internal
    port: 5672
    username: ${RABBITMQ_USER}
    password: ${RABBITMQ_PASS}
    virtual-host: /prod
    connection-timeout: 5000
    publisher-confirm-type: correlated
    publisher-returns: true
    cache:
      channel:
        size: 50
        checkout-timeout: 5000
      connection:
        mode: channel
    listener:
      simple:
        acknowledge-mode: manual
        prefetch: 100
        concurrency: 5
        max-concurrency: 20
        default-requeueable: false   # 默认不重新入队
        missing-queues-fatal: false  # 队列不存在不报错
```

## 三、代码规范

```java
// 1. 统一消息信封
public class MessageEnvelope<T> {
    private String messageId = UUID.randomUUID().toString();
    private String type;
    private Long timestamp = System.currentTimeMillis();
    private String traceId = TraceContext.getTraceId();
    private String source = "order-service";
    private T payload;
    private Map<String, String> headers = new HashMap<>();
}

// 2. 统一 Exchange/Queue 常量
public class MQConstants {
    // Exchange
    public static final String ORDER_EXCHANGE = "order-exchange";
    public static final String DLX_EXCHANGE = "dlx-exchange";

    // Queue
    public static final String ORDER_QUEUE = "order-queue";
    public static final String ORDER_DLQ = "order-queue-dlq";

    // Routing Key
    public static final String ORDER_CREATED = "order.created";
    public static final String ORDER_PAID = "order.paid";

    // 参数
    public static final int MAX_PRIORITY = 10;
    public static final long TTL_30MIN = 1800000;
}

// 3. 统一异常处理
@Component
public class MQErrorHandler {

    public void handleConsumeError(Message message, Channel channel,
                                    Exception e) throws IOException {
        long tag = message.getMessageProperties().getDeliveryTag();
        int retryCount = getRetryCount(message);

        if (retryCount >= MAX_RETRY) {
            // 超过重试次数，进入死信队列
            channel.basicNack(tag, false, false);
            alertService.alert("消费失败进入死信队列", message, e);
        } else {
            // 重新入队重试
            channel.basicNack(tag, false, true);
        }
    }

    private int getRetryCount(Message message) {
        List<Map<String, ?>> xDeath = message.getMessageProperties()
            .getHeaders().get("x-death");
        if (xDeath == null || xDeath.isEmpty()) return 0;
        return ((Number) xDeath.get(0).get("count")).intValue();
    }
}
```

## 四、监控告警

```yaml
# Prometheus 告警规则
groups:
  - name: rabbitmq
    rules:
      - alert: RabbitMQConsumerLagHigh
        expr: rabbitmq_queue_messages - rabbitmq_queue_messages_ready > 10000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "队列 {{ $labels.queue }} 消费积压"

      - alert: RabbitMQMemoryHigh
        expr: rabbitmq_process_memory_bytes / node_memory_MemTotal_bytes > 0.7
        for: 5m
        labels:
          severity: critical

      - alert: RabbitMQDiskLow
        expr: rabbitmq_disk_free_bytes < 2147483648
        for: 1m
        labels:
          severity: critical
```

## 五、容灾方案

```java
// 1. 本地消息表 - 保证消息不丢
@Service
public class ReliableMessageService {

    @Transactional
    public void sendReliably(String topic, Object message) {
        // 数据库操作
        orderMapper.insert(order);
        // 保存待发送消息
        messageMapper.insert(new PendingMessage(topic, JSON.toJSONString(message)));
    }

    @Scheduled(fixedDelay = 1000)
    public void pollAndSend() {
        List<PendingMessage> pending = messageMapper.selectPending(100);
        for (PendingMessage msg : pending) {
            try {
                rabbitTemplate.convertAndSend(msg.getTopic(), msg.getContent());
                messageMapper.markSent(msg.getId());
            } catch (Exception e) {
                messageMapper.incrementRetry(msg.getId());
            }
        }
    }
}

// 2. 消费幂等 - 保证不重复消费
@Service
public class IdempotentConsumer {
    public boolean tryProcess(String messageId) {
        return redisTemplate.opsForValue()
            .setIfAbsent("mq:idempotent:" + messageId, "1", 24, TimeUnit.HOURS);
    }
}
```

## 六、运维检查清单

```
部署前检查:
  □ Erlang 版本与 RabbitMQ 版本匹配
  □ 集群节点 Erlang Cookie 一致
  □ 修改默认密码
  □ 配置持久化（队列 + 消息）
  □ 配置死信队列
  □ 配置监控告警

上线检查:
  □ 消费者手动 ACK
  □ 生产者 Publisher Confirm
  □ Prefetch 合理设置
  □ 队列长度限制
  □ 网络分区策略
  □ 备份策略
```

## 七、常见问题排查

```bash
# 消息堆积 - 查看队列消息数
rabbitmqctl list_queues name messages consumers

# 连接泄漏 - 查看连接数
rabbitmqctl list_connections

# 内存不足 - 查看内存使用
rabbitmqctl status | grep memory

# 网络分区 - 查看集群状态
rabbitmqctl cluster_status
```

## 八、注意事项

1. **不要在 @RabbitListener 中做耗时操作**，异步处理或放入线程池
2. **Exchange/Queue 声明放在配置类中**，不要散落在各个消费者里
3. **版本升级前做好测试**，RabbitMQ 大版本升级可能有不兼容变化
4. **定期清理未使用的队列和 Exchange**，避免元数据膨胀
5. **监控消费延迟是最有效的预警手段**，比消息数量更重要
