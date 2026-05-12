# Kafka vs RocketMQ

## 一、核心对比

| 维度 | Kafka | RocketMQ |
|------|-------|----------|
| 吞吐量 | 百万级 | 十万级 |
| 事务消息 | 0.11+ 支持 | 原生支持 |
| 延迟消息 | 不支持 | 18 级别 |
| 消息过滤 | 不支持 | Tag/SQL92 |
| 消息轨迹 | 需额外实现 | 内置 |
| 消息回溯 | 按 Offset | 按时间/Offset |
| 消费模式 | 拉模式 | 推/拉 |
| 社区 | 国际化 | 国内为主 |

## 二、选型建议

```yaml
选 Kafka:
  - 日志收集、大数据管道
  - 流处理（Kafka Streams/Flink）
  - 超高吞吐需求

选 RocketMQ:
  - 电商/金融交易
  - 需要事务消息
  - 需要延迟消息
  - 需要消息轨迹
  - 国内团队运维
```

## 三、架构对比

```
Kafka 架构:
  Broker (ZooKeeper/KRaft) → Topic → Partition → Replica
  消费者组通过 Offset 追踪消费进度
  数据按 Partition 分片，每分区有 Leader + Follower

RocketMQ 架构:
  NameServer → Broker (Master-Slave) → Topic → Queue
  消费者组通过 ConsumeOffset 追踪消费进度
  支持 Broker 端消息过滤 (Tag / SQL92)
```

## 四、生态对比

```
维度                  Kafka                         RocketMQ
流处理                Kafka Streams, Flink           RocketMQ Streams
连接器                Kafka Connect (丰富)           RocketMQ Connect (较少)
SQL 查询              ksqlDB                         无
Schema 管理           Schema Registry (Confluent)    无内置方案
监控工具              AKHQ, Confluent Control Center  RocketMQ Dashboard
云服务                AWS MSK, 阿里云 Kafka           阿里云 RocketMQ
客户端语言            Java, Python, Go, C++          Java 为主
```

## 五、消息过滤对比

```java
// RocketMQ Tag 过滤（Broker 端）
consumer.subscribe("OrderTopic", "TagA || TagB");

// RocketMQ SQL92 过滤
consumer.subscribe("OrderTopic",
    MessageSelector.bySql("amount > 100 AND region = 'CN'"));

// Kafka 不支持 Broker 端过滤，需在消费端实现
@KafkaListener(topics = "order-events")
public void consume(ConsumerRecord<String, String> record) {
    Order order = JSON.parseObject(record.value(), Order.class);
    if (order.getAmount().compareTo(BigDecimal.valueOf(100)) > 0) {
        process(order);  // 应用层过滤
    }
}
```

## 六、注意事项

1. **Kafka 吞吐量更高**，但 RocketMQ 功能更丰富
2. **RocketMQ 事务消息是核心优势**
3. **Kafka 的生态系统更完善**（Connect、Streams、ksqlDB）
4. **RocketMQ 中文文档和社区更好**
5. **性能差距在合理配置下不大**，功能需求更重要
6. **Kafka KRaft 模式去除 ZooKeeper 依赖**，运维简化
7. **RocketMQ 5.0 新增 Pop 消费模式和轻量级客户端**
