# Exactly-Once 语义

## 一、三种语义对比

```
At-most-once:  最多一次 - 可能丢，不重复
At-least-once: 至少一次 - 不丢，可能重复
Exactly-once:  精确一次 - 不丢，不重复
```

## 二、Kafka Exactly-once

```java
// 幂等生产者 + 事务
Properties props = new Properties();
props.put("enable.idempotence", true);
props.put("transactional.id", "my-transactional-id");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();

// 消费-处理-生产的 Exactly-once
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    producer.beginTransaction();
    try {
        for (ConsumerRecord<String, String> record : records) {
            String result = process(record.value());
            producer.send(new ProducerRecord<>("output-topic", record.key(), result));
        }
        // 提交消费 Offset 到事务
        producer.sendOffsetsToTransaction(
            currentOffsets(consumer), consumer.groupMetadata());
        producer.commitTransaction();
    } catch (Exception e) {
        producer.abortTransaction();
    }
}
```

## 三、RocketMQ Exactly-once

```java
// 事务消息 + 消费幂等
rocketMQTemplate.sendMessageInTransaction("Topic", msg, arg);

// 消费端幂等
@RocketMQMessageListener(topic = "Topic")
public class Consumer implements RocketMQListener<Message> {
    @Override
    public void onMessage(Message msg) {
        String id = msg.getKeys();
        if (redisTemplate.opsForValue().setIfAbsent("eo:" + id, "1", 24, TimeUnit.HOURS)) {
            process(msg);
        }
    }
}
```

## 四、跨系统 Exactly-once

```java
// Kafka 内部 Exactly-once 不等于外部系统
// 外部系统需要配合事务或幂等

// 方案1: 事务性写入外部系统
@Transactional
public void process(ConsumerRecord<String, String> record) {
    // 数据库和 Offset 在同一事务中
    orderMapper.insert(order);
    offsetMapper.saveOffset(record);
}

// 方案2: 幂等写入外部系统
public void process(ConsumerRecord<String, String> record) {
    String idempotentKey = record.topic() + ":" + record.partition() + ":" + record.offset();
    if (!offsetMapper.exists(idempotentKey)) {
        orderMapper.insert(order);
        offsetMapper.save(idempotentKey);
    }
}
```

## 五、注意事项

1. **Exactly-once 只在 Kafka 内部生效**，外部系统需要自己保证
2. **事务消息有 3-5% 的性能开销**
3. **幂等是实现 Exactly-once 最实用的方案**
4. **跨系统的 Exactly-once 需要分布式事务配合**
5. **EXACTLY_ONCE_V2（Kafka 3.0+）性能优于 V1**
