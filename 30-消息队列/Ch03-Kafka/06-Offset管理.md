# Offset 管理

## 一、Offset 存储机制

Kafka 0.9+ 将 Consumer Group 的 Offset 存储在内部 Topic `__consumer_offsets` 中，而非 ZooKeeper。

```
__consumer_offsets (内部 Topic)
├── [order-group, order-events, 0]  -> offset=1000
├── [order-group, order-events, 1]  -> offset=2000
├── [order-group, order-events, 2]  -> offset=1500
└── [payment-group, payment-events, 0] -> offset=500
```

## 二、自动提交 vs 手动提交

```java
// 自动提交 - 默认每 5 秒提交
props.put("enable.auto.commit", true);
props.put("auto.commit.interval.ms", 5000);
// 问题: 可能提交了未处理的 Offset，崩溃后消息丢失

// 手动同步提交 - 处理完再提交（推荐）
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        process(record);
    }
    consumer.commitSync();
}

// 手动异步提交 - 高性能
consumer.commitAsync((offsets, exception) -> {
    if (exception != null) {
        log.error("提交失败，下次重试: {}", exception.getMessage());
    }
});
```

## 三、Offset 重置

```java
// auto.offset.reset 策略
// earliest: 从最早的 Offset 开始
// latest:   从最新的 Offset 开始（默认）
// none:     没有 Offset 时报错

props.put("auto.offset.reset", "earliest");
```

```bash
# 命令行重置 Offset
# 重置到最早
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-group --topic order-events \
  --reset-offsets --to-earliest --execute

# 重置到最新
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-group --topic order-events \
  --reset-offsets --to-latest --execute

# 重置到指定 Offset
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-group --topic order-events \
  --reset-offsets --to-offset 1000 --execute

# 重置到指定时间
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group my-group --topic order-events \
  --reset-offsets --to-datetime 2024-01-01T00:00:00.000 --execute

# 重置需要先让 Group 停止消费（无活跃消费者）
```

## 四、手动管理 Offset

```java
// 按消息粒度提交 Offset
public void consumeWithManualOffset() {
    Map<TopicPartition, OffsetAndMetadata> currentOffsets = new HashMap<>();

    while (true) {
        ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
        int count = 0;
        for (ConsumerRecord<String, String> record : records) {
            process(record);

            // 记录 Offset
            currentOffsets.put(
                new TopicPartition(record.topic(), record.partition()),
                new OffsetAndMetadata(record.offset() + 1)
            );

            // 每 1000 条提交一次
            if (++count % 1000 == 0) {
                consumer.commitSync(currentOffsets);
            }
        }
        // 提交剩余 Offset
        if (!currentOffsets.isEmpty()) {
            consumer.commitSync(currentOffsets);
        }
    }
}
```

## 五、查看 Offset 状态

```bash
# 查看消费者组 Offset
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group my-group

# 输出:
# TOPIC          PARTITION  CURRENT-OFFSET  LOG-END-OFFSET  LAG
# order-events   0          10000           10050           50
# order-events   1          20000           20000           0
# order-events   2          15000           15200           200

# 查看 Topic 的最早/最新 Offset
kafka-run-class kafka.tools.GetOffsetShell \
  --broker-list localhost:9092 \
  --topic order-events \
  --time -1  # -1=最新, -2=最早
```

## 六、注意事项

1. **先处理消息再提交 Offset**，否则消息可能丢失
2. **Offset 重置需要先停止消费者组**，否则不生效
3. **__consumer_offsets 是内部 Topic**，不要手动修改
4. **异步提交可能有乱序**，确保只在处理完所有消息后提交
5. **重复消费比丢失消息好**，设计时优先保证至少一次语义
