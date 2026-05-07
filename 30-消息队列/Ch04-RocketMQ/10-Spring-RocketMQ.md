# Spring Boot 集成 RocketMQ

## 一、依赖配置

```xml
<dependency>
    <groupId>org.apache.rocketmq</groupId>
    <artifactId>rocketmq-spring-boot-starter</artifactId>
    <version>2.2.3</version>
</dependency>
```

```yaml
rocketmq:
  name-server: localhost:9876
  producer:
    group: my-producer-group
    send-message-timeout: 3000
    retry-times-when-send-failed: 2
  consumer:
    group: my-consumer-group
    topic: OrderTopic
```

## 二、生产者

```java
@Service
public class OrderProducerService {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    // 同步发送
    public void sendSync(Order order) {
        rocketMQTemplate.syncSend("OrderTopic", order);
    }

    // 异步发送
    public void sendAsync(Order order) {
        rocketMQTemplate.asyncSend("OrderTopic", order, new SendCallback() {
            @Override
            public void onSuccess(SendResult result) { }
            @Override
            public void onException(Throwable e) { }
        });
    }

    // 顺序发送
    public void sendOrderly(Order order) {
        rocketMQTemplate.syncSendOrderly("OrderTopic", order, order.getOrderId());
    }

    // 延迟发送
    public void sendDelay(Order order, int level) {
        Message<Order> msg = MessageBuilder.withPayload(order).build();
        rocketMQTemplate.syncSend("DelayTopic", msg, 3000, level);
    }

    // 事务发送
    public void sendTransaction(Order order) {
        rocketMQTemplate.sendMessageInTransaction(
            "OrderTopic",
            MessageBuilder.withPayload(order).build(),
            order
        );
    }
}
```

## 三、消费者

```java
@Component
@RocketMQMessageListener(
    topic = "OrderTopic",
    consumerGroup = "OrderConsumerGroup",
    selectorExpression = "*",
    consumeMode = ConsumeMode.CONCURRENTLY,
    messageModel = MessageModel.CLUSTERING
)
public class OrderConsumer implements RocketMQListener<Order> {

    @Override
    public void onMessage(Order order) {
        log.info("收到订单: {}", order.getId());
        processOrder(order);
    }
}

// 带 ACK 的消费
@Component
@RocketMQMessageListener(topic = "OrderTopic", consumerGroup = "AckGroup")
public class AckConsumer implements RocketMQListener<MessageExt> {

    @Override
    public void onMessage(MessageExt msg) {
        try {
            process(msg);
        } catch (Exception e) {
            // 处理失败，稍后重试
            throw new RuntimeException("消费失败", e);
        }
    }
}
```

## 四、消息轨迹

```yaml
rocketmq:
  producer:
    group: my-producer-group
    enable-msg-trace: true         # 开启消息轨迹
    customized-trace-topic: RMQ_SYS_TRACE_TOPIC
```

## 五、多 Topic 配置

```java
// 多 Topic 生产者
@Service
public class MultiTopicProducer {

    @Autowired
    private RocketMQTemplate rocketMQTemplate;

    public void sendToDifferentTopics(Order order, Payment payment) {
        rocketMQTemplate.syncSend("OrderTopic", order);
        rocketMQTemplate.syncSend("PaymentTopic", payment);
    }
}

// 多消费者
@Component
@RocketMQMessageListener(topic = "OrderTopic", consumerGroup = "Group1")
public class Consumer1 implements RocketMQListener<Order> { }

@Component
@RocketMQMessageListener(topic = "PaymentTopic", consumerGroup = "Group2")
public class Consumer2 implements RocketMQListener<Payment> { }
```

## 六、注意事项

1. **RocketMQTemplate 是线程安全的**，可以全局共享
2. **consumer-group 命名要全局唯一**
3. **selectorExpression 默认是 * **，订阅所有 Tag
4. **开启消息轨迹会增加 Broker 负载**
5. **Consumer 类必须实现 RocketMQListener 接口**
