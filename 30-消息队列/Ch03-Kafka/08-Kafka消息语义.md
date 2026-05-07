# Kafka 消息语义

## 一、三种消息语义

```
语义              生产者                  消费者
-----------------------------------------------------------
At-most-once     可能丢，不重复          可能丢，不重复
At-least-once    不丢，可能重复          不丢，可能重复
Exactly-once     不丢，不重复            不丢，不重复
```

## 二、At-least-once（至少一次）

```java
// 生产者: 重试 + acks=all
props.put("acks", "all");
props.put("retries", Integer.MAX_VALUE);

// 消费者: 先处理再提交 Offset
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        processMessage(record);  // 先处理
    }
    consumer.commitSync();       // 再提交
}
// 问题: 处理完但提交前崩溃 -> 重复消费
```

## 三、At-most-once（最多一次）

```java
// 消费者: 先提交再处理
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    consumer.commitSync();       // 先提交
    for (ConsumerRecord<String, String> record : records) {
        processMessage(record);  // 再处理
    }
}
// 问题: 提交后处理前崩溃 -> 消息丢失
```

## 四、Exactly-once（精确一次）

### 4.1 幂等生产者

```java
// Kafka 0.11+ 幂等生产者
props.put("enable.idempotence", true);
// 自动设置: acks=all, retries=MAX, max.in.flight.requests=5

// 幂等保证: 同一条消息无论发送多少次，结果相同
// 原理: ProducerID + SequenceNumber 去重
```

### 4.2 事务消息

```java
// 跨分区原子写入
props.put("transactional.id", "my-transactional-id");
props.put("enable.idempotence", true);

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
producer.initTransactions();

try {
    producer.beginTransaction();

    // 原子写入多个分区
    producer.send(new ProducerRecord<>("topic-a", "key", "value1"));
    producer.send(new ProducerRecord<>("topic-b", "key", "value2"));

    // 提交消费 Offset（消费-处理-生产的 Exactly-once）
    producer.sendOffsetsToTransaction(offsets, consumerGroupId);

    producer.commitTransaction();
} catch (Exception e) {
    producer.abortTransaction();
}
```

### 4.3 Kafka Streams Exactly-once

```java
// Kafka Streams 内置 Exactly-once
StreamsConfig props = new StreamsConfig();
props.put(StreamsConfig.PROCESSING_GUARANTEE_CONFIG,
    StreamsConfig.EXACTLY_ONCE_V2);  // Exactly-once 语义

StreamsBuilder builder = new StreamsBuilder();
builder.<String, String>stream("input-topic")
    .mapValues(value -> process(value))
    .to("output-topic");

// Kafka Streams 内部使用事务保证 Exactly-once
```

## 五、消费-处理-生产模式

```java
// 完整的 Exactly-once 流程
public void consumeProcessProduce() {
    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));

        producer.beginTransaction();
        try {
            for (ConsumerRecord<String, String> record : records) {
                // 处理
                String result = process(record.value());
                // 生产
                producer.send(new ProducerRecord<>("output-topic", record.key(), result));
            }

            // 提交消费 Offset 到事务
            Map<TopicPartition, OffsetAndMetadata> offsets = buildOffsets(records);
            producer.sendOffsetsToTransaction(offsets, consumer.groupMetadata());

            producer.commitTransaction();
        } catch (Exception e) {
            producer.abortTransaction();
            resetToLastCommittedOffset();
        }
    }
}
```

## 六、注意事项

1. **幂等生产者只保证单分区 Exactly-once**，跨分区需要事务
2. **事务消息有性能开销**，大约 3-5% 的吞吐量下降
3. **Exactly-once 只在 Kafka 内部生效**，外部系统需要自己保证
4. **消费端的 Exactly-once 需要幂等处理**
5. **EXACTLY_ONCE_V2 比 V1 性能更好**，Kafka 3.0+ 推荐
