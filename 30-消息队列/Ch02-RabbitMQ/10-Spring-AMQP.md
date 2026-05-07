# Spring AMQP

## 一、依赖配置

```xml
<!-- pom.xml -->
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-amqp</artifactId>
</dependency>
```

```yaml
# application.yml
spring:
  rabbitmq:
    host: localhost
    port: 5672
    username: admin
    password: admin123
    virtual-host: /
    connection-timeout: 5000
    publisher-confirm-type: correlated    # 生产者确认
    publisher-returns: true               # 消息退回
    listener:
      simple:
        acknowledge-mode: manual          # 手动确认
        prefetch: 100                     # 预取数量
        concurrency: 3                    # 并发消费者数
        max-concurrency: 10               # 最大并发数
        retry:
          enabled: true
          max-attempts: 3
          initial-interval: 1000
```

## 二、配置类

```java
@Configuration
@EnableRabbit
public class RabbitMQConfig {

    // 交换机
    @Bean
    public TopicExchange orderExchange() {
        return new TopicExchange("order-exchange", true, false);
    }

    // 队列
    @Bean
    public Queue orderQueue() {
        return QueueBuilder.durable("order-queue")
            .quorum()
            .withArgument("x-dead-letter-exchange", "dlx-exchange")
            .build();
    }

    // 绑定
    @Bean
    public Binding orderBinding() {
        return BindingBuilder.bind(orderQueue())
            .to(orderExchange())
            .with("order.#");
    }

    // JSON 消息转换器
    @Bean
    public MessageConverter jsonMessageConverter() {
        return new Jackson2JsonMessageConverter();
    }

    // RabbitTemplate 配置
    @Bean
    public RabbitTemplate rabbitTemplate(ConnectionFactory factory) {
        RabbitTemplate template = new RabbitTemplate(factory);
        template.setMessageConverter(jsonMessageConverter());
        template.setMandatory(true);

        // Confirm 回调
        template.setConfirmCallback((correlationData, ack, cause) -> {
            if (!ack) {
                log.error("消息发送失败: {}", cause);
            }
        });

        // Return 回调
        template.setReturnsCallback(returned -> {
            log.error("消息路由失败: {}", returned.getMessage());
        });

        return template;
    }
}
```

## 三、生产者

```java
@Service
public class OrderProducer {

    @Autowired
    private RabbitTemplate rabbitTemplate;

    // 简单发送
    public void sendOrder(Order order) {
        rabbitTemplate.convertAndSend(
            "order-exchange",
            "order.created",
            order
        );
    }

    // 带 CorrelationData 的发送
    public void sendOrderWithConfirm(Order order) {
        CorrelationData correlationData = new CorrelationData(order.getId().toString());
        rabbitTemplate.convertAndSend(
            "order-exchange",
            "order.created",
            order,
            correlationData
        );
    }

    // 延迟发送
    public void sendDelayedOrder(Order order, long delayMs) {
        rabbitTemplate.convertAndSend(
            "delayed-exchange",
            "order.delay",
            order,
            message -> {
                message.getMessageProperties().setHeader("x-delay", delayMs);
                return message;
            }
        );
    }
}
```

## 四、消费者

```java
@Component
public class OrderConsumer {

    // 基本消费
    @RabbitListener(queues = "order-queue")
    public void handleOrder(Order order) {
        log.info("收到订单: {}", order.getId());
        orderService.process(order);
    }

    // 手动 ACK
    @RabbitListener(queues = "order-queue")
    public void handleOrderWithAck(Message message, Channel channel) throws IOException {
        long tag = message.getMessageProperties().getDeliveryTag();
        try {
            Order order = (Order) messageConverter.fromMessage(message);
            processOrder(order);
            channel.basicAck(tag, false);
        } catch (Exception e) {
            channel.basicNack(tag, false, true);
        }
    }

    // 获取消息头信息
    @RabbitListener(queues = "order-queue")
    public void handleWithHeaders(
        @Payload Order order,
        @Header("amqp_deliveryTag") long tag,
        @Header("amqp_redelivered") boolean redelivered,
        Channel channel
    ) throws IOException {
        processOrder(order);
        channel.basicAck(tag, false);
    }

    // 动态队列名
    @RabbitListener(queues = "#{'${mq.order.queue}'}")
    public void handleDynamicQueue(Order order) {
        processOrder(order);
    }
}
```

## 五、批量消费

```java
@Configuration
public class BatchConfig {

    @Bean
    public SimpleRabbitListenerContainerFactory batchListenerFactory() {
        SimpleRabbitListenerContainerFactory factory = new SimpleRabbitListenerContainerFactory();
        factory.setConnectionFactory(connectionFactory());
        factory.setBatchListener(true);            // 开启批量
        factory.setConsumerBatchEnabled(true);
        factory.setBatchSize(50);                  // 批量大小
        factory.setReceiveTimeout(5000L);          // 接收超时
        factory.setMessageConverter(new Jackson2JsonMessageConverter());
        return factory;
    }
}

// 批量消费者
@RabbitListener(queues = "order-queue", containerFactory = "batchListenerFactory")
public void handleBatch(List<Order> orders) {
    log.info("批量处理 {} 条订单", orders.size());
    orderService.batchProcess(orders);
}
```

## 六、注意事项

1. **使用 @RabbitListener 注解比手动创建消费者更简洁**
2. **JSON 转换器要配置为全局 Bean**，否则消息格式不一致
3. **手动 ACK 时 Channel 对象不要缓存**，每个线程有独立 Channel
4. **批量消费注意单次处理时间**，避免超时触发 Rebalance
5. **@Header 注解获取 AMQP 内置头信息**，自定义头用 `@Headers Map` 获取
