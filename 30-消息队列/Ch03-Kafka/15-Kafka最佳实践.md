# Kafka 最佳实践

## 一、Topic 设计规范

```yaml
Topic 命名:
  格式: {业务域}-{数据类型}-{环境}
  示例:
    - order-events-prod
    - user-behavior-log
    - payment-callback-dev

分区规划:
  - 保守: 预期消费者数 * 2
  - 激进: 目标吞吐量 / 单分区吞吐量
  - 上限: 单 Topic 不超过 100 分区

副本配置:
  生产环境: replication-factor=3, min.insync.replicas=2
  开发环境: replication-factor=1
```

## 二、生产者最佳实践

```java
// 推荐配置
Properties props = new Properties();
props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");
props.put("acks", "all");                    // 最高可靠性
props.put("retries", Integer.MAX_VALUE);     // 无限重试
props.put("enable.idempotence", true);       // 幂等
props.put("max.in.flight.requests.per.connection", 5);
props.put("compression.type", "snappy");     // 压缩
props.put("batch.size", 32768);              // 32KB
props.put("linger.ms", 5);                   // 5ms
props.put("buffer.memory", 67108864);        // 64MB

// 消息设计
public class MessageEnvelope<T> {
    private String messageId = UUID.randomUUID().toString();
    private String type;          // 消息类型
    private Long timestamp = System.currentTimeMillis();
    private String traceId;       // 链路追踪
    private T payload;
}
```

## 三、消费者最佳实践

```java
// 推荐配置
Properties props = new Properties();
props.put("bootstrap.servers", "kafka1:9092,kafka2:9092,kafka3:9092");
props.put("group.id", "my-service-group");
props.put("enable.auto.commit", false);       // 手动提交
props.put("auto.offset.reset", "earliest");   // 无 Offset 从头消费
props.put("max.poll.records", 500);           // 每次最多 500 条
props.put("max.poll.interval.ms", 300000);    // 处理超时 5 分钟
props.put("session.timeout.ms", 10000);       // 会话超时 10 秒
props.put("partition.assignment.strategy",
    "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");

// 消费逻辑
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    try {
        for (ConsumerRecord<String, String> record : records) {
            processWithIdempotent(record);
        }
        consumer.commitSync();
    } catch (Exception e) {
        log.error("处理失败", e);
        // 不提交，下次重新消费
    }
}
```

## 四、可靠性保障

```java
// 生产端: 本地消息表
@Transactional
public void saveAndSend(Order order) {
    orderMapper.insert(order);
    pendingMessageMapper.insert(new PendingMessage(order));
}

@Scheduled(fixedDelay = 1000)
public void retryPending() {
    List<PendingMessage> pending = pendingMessageMapper.selectPending(100);
    for (PendingMessage msg : pending) {
        kafkaTemplate.send(msg.getTopic(), msg.getContent())
            .whenComplete((result, ex) -> {
                if (ex == null) pendingMessageMapper.markSent(msg.getId());
            });
    }
}

// 消费端: 幂等处理
public boolean processWithIdempotent(ConsumerRecord<String, String> record) {
    String messageId = extractMessageId(record);
    if (!redisTemplate.opsForValue().setIfAbsent("kafka:idem:" + messageId, "1", 24, TimeUnit.HOURS)) {
        return false;  // 已处理
    }
    try {
        doProcess(record);
        return true;
    } catch (Exception e) {
        redisTemplate.delete("kafka:idem:" + messageId);
        throw e;
    }
}
```

## 五、运维最佳实践

```yaml
集群规划:
  - 至少 3 个 Broker（推荐 3-5 个）
  - 跨机架/跨可用区部署副本
  - Controller 节点与 Broker 节点分离（大规模）

数据保留:
  - 日志保留: 7 天 (retention.ms=604800000)
  - 日志压缩: 对于更新类 Topic 开启
  - 分段大小: 1GB (segment.bytes=1073741824)

监控告警:
  - UnderReplicatedPartitions > 0 -> 严重告警
  - Consumer Lag > 100000 -> 警告告警
  - ActiveControllerCount != 1 -> 严重告警
  - 磁盘使用率 > 80% -> 警告告警
```

## 六、常见问题排查

```bash
# 消费积压
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group lag-group

# 副本状态
kafka-topics --bootstrap-server localhost:9092 \
  --describe --under-replicated-partitions

# Topic 磁盘使用
kafka-log-dirs --bootstrap-server localhost:9092 \
  --describe --topic-list order-events
```

## 七、注意事项

1. **生产环境务必 acks=all + min.insync.replicas=2**
2. **消费者组名要语义化**，不要使用随机名称
3. **分区数只增不减**，规划时预留扩展空间
4. **KRaft 模式是 3.0+ 的首选**，新集群不要再用 ZooKeeper
5. **Topic 清理需要谨慎**，删除后数据不可恢复
6. **不要在 Kafka 中存储大消息**，建议不超过 1MB
7. **跨数据中心同步用 MirrorMaker 2**，支持双向复制
