# Spring Kafka

## 一、依赖配置

```xml
<dependency>
    <groupId>org.springframework.kafka</groupId>
    <artifactId>spring-kafka</artifactId>
</dependency>
```

```yaml
spring:
  kafka:
    bootstrap-servers: localhost:9092
    producer:
      key-serializer: org.apache.kafka.common.serialization.StringSerializer
      value-serializer: org.apache.kafka.common.serialization.StringSerializer
      acks: all
      properties:
        enable.idempotence: true
        retries: 2147483647
    consumer:
      group-id: my-group
      key-deserializer: org.apache.kafka.common.serialization.StringDeserializer
      value-deserializer: org.apache.kafka.common.serialization.StringDeserializer
      auto-offset-reset: earliest
      enable-auto-commit: false
      properties:
        max.poll.records: 100
    listener:
      ack-mode: manual_immediate
      concurrency: 5
```

## 二、生产者配置

```java
@Configuration
public class KafkaProducerConfig {

    @Bean
    public ProducerFactory<String, Object> producerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, StringSerializer.class);
        props.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, JsonSerializer.class);
        props.put(ProducerConfig.ACKS_CONFIG, "all");
        props.put(ProducerConfig.ENABLE_IDEMPOTENCE_CONFIG, true);
        return new DefaultKafkaProducerFactory<>(props);
    }

    @Bean
    public KafkaTemplate<String, Object> kafkaTemplate() {
        return new KafkaTemplate<>(producerFactory());
    }
}

// 使用
@Service
public class OrderService {
    @Autowired
    private KafkaTemplate<String, Object> kafkaTemplate;

    // 同步发送
    public void sendOrder(Order order) {
        kafkaTemplate.send("order-events", order.getId(), order);
    }

    // 异步发送带回调
    public void sendWithCallback(Order order) {
        CompletableFuture<SendResult<String, Object>> future =
            kafkaTemplate.send("order-events", order.getId(), order);

        future.whenComplete((result, ex) -> {
            if (ex != null) {
                log.error("发送失败", ex);
            } else {
                log.info("发送成功: {}", result.getRecordMetadata());
            }
        });
    }

    // 事务发送
    @Transactional
    public void sendTransactional(Order order) {
        orderMapper.insert(order);
        kafkaTemplate.send("order-events", order.getId(), order);
    }
}
```

## 三、消费者配置

```java
@Configuration
public class KafkaConsumerConfig {

    @Bean
    public ConsumerFactory<String, Object> consumerFactory() {
        Map<String, Object> props = new HashMap<>();
        props.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(ConsumerConfig.GROUP_ID_CONFIG, "my-group");
        props.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, StringDeserializer.class);
        props.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, JsonDeserializer.class);
        props.put(JsonDeserializer.TRUSTED_PACKAGES, "com.example.*");
        props.put(ConsumerConfig.ENABLE_AUTO_COMMIT_CONFIG, false);
        return new DefaultKafkaConsumerFactory<>(props);
    }

    @Bean
    public ConcurrentKafkaListenerContainerFactory<String, Object> kafkaListenerContainerFactory() {
        ConcurrentKafkaListenerContainerFactory<String, Object> factory =
            new ConcurrentKafkaListenerContainerFactory<>();
        factory.setConsumerFactory(consumerFactory());
        factory.setConcurrency(5);
        factory.getContainerProperties().setAckMode(ContainerProperties.AckMode.MANUAL_IMMEDIATE);
        return factory;
    }
}
```

## 四、消费者使用

```java
@Component
public class OrderConsumer {

    // 基本消费
    @KafkaListener(topics = "order-events", groupId = "order-group")
    public void consume(Order order) {
        log.info("收到订单: {}", order.getId());
    }

    // 手动 ACK
    @KafkaListener(topics = "order-events")
    public void consumeWithAck(ConsumerRecord<String, Object> record,
                               Acknowledgment ack) {
        process(record);
        ack.acknowledge();
    }

    // 批量消费
    @KafkaListener(topics = "order-events", containerFactory = "batchFactory")
    public void batchConsume(List<ConsumerRecord<String, Object>> records) {
        for (ConsumerRecord<String, Object> record : records) {
            process(record);
        }
    }

    // 获取消息头
    @KafkaListener(topics = "order-events")
    public void consumeWithHeaders(
        @Payload Order order,
        @Header(KafkaHeaders.RECEIVED_PARTITION) int partition,
        @Header(KafkaHeaders.OFFSET) long offset,
        @Header(KafkaHeaders.RECEIVED_KEY) String key
    ) {
        log.info("partition={}, offset={}, key={}, order={}", partition, offset, key, order);
    }

    // 条件消费
    @KafkaListener(topics = "order-events", properties = {
        "filter.headers:orderType"
    })
    @ConditionalOnProperty(name = "kafka.consumer.orders.enabled", havingValue = "true")
    public void conditionalConsume(Order order) {
        process(order);
    }
}
```

## 五、Topic 管理

```java
@Configuration
public class TopicConfig {

    @Bean
    public NewTopic orderEventsTopic() {
        return TopicBuilder.name("order-events")
            .partitions(6)
            .replicas(3)
            .config("retention.ms", "604800000")
            .config("compression.type", "snappy")
            .build();
    }

    @Bean
    public NewTopic orderDlqTopic() {
        return TopicBuilder.name("order-events-dlq")
            .partitions(3)
            .replicas(3)
            .build();
    }
}
```

## 六、注意事项

1. **JsonDeserializer 要配置 TRUSTED_PACKAGES**，否则反序列化失败
2. **@Transactional 需要配合 KafkaTransactionManager** 才能事务化
3. **批量消费要配置 BatchMessagingMessageConverter**
4. **手动 ACK 模式下要在处理完成后调用 ack.acknowledge()**
5. **并发数不能超过分区数**，否则多余的消费者空闲
