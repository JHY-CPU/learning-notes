# RabbitMQ vs Kafka

## 一、架构对比

```
RabbitMQ: AMQP 协议，Exchange -> Queue -> Consumer
Kafka:    分布式日志，Topic -> Partition -> Consumer Group
```

## 二、核心对比

| 维度 | RabbitMQ | Kafka |
|------|----------|-------|
| 吞吐量 | 万级/秒 | 百万级/秒 |
| 延迟 | 微秒级 | 毫秒级 |
| 消息持久化 | 内存+磁盘 | 磁盘顺序写 |
| 消息回溯 | 不支持 | 支持 |
| 路由灵活性 | 高（4种Exchange） | 低（仅Topic） |
| 消息顺序 | 队列有序 | 分区有序 |
| 事务消息 | 不支持 | 支持 |
| 消费模式 | 推/拉 | 拉 |
| 开发语言 | Erlang | Java/Scala |

## 三、适用场景

```yaml
RabbitMQ 适用:
  - 业务消息（订单、支付通知）
  - 复杂路由需求
  - 低延迟场景
  - 小规模系统

Kafka 适用:
  - 日志收集与分析
  - 大数据管道
  - 流处理
  - 高吞吐场景
  - 事件溯源
```

## 四、消息可靠性对比

```
维度                RabbitMQ                    Kafka
生产者确认           publisher-confirm           acks=all + retries
消费者确认           manual-ack / auto-ack       手动提交 Offset
消息持久化           durable queue + mirrored    多副本 (replication.factor)
丢消息风险           内存队列可能丢               acks=1 时可能丢
重复消费             可能（手动 ack 丢失）         可能（Offset 提交失败）
顺序保证             单队列有序                   分区内有序
```

## 五、集群与运维对比

```
维度                RabbitMQ                    Kafka
集群方式             镜像队列 / Quorum Queue     分区副本机制
扩缩容               手动迁移队列                 自动 Rebalance
管理工具             Management UI               Kafka Manager / AKHQ
监控指标             队列深度、消费者速率          Lag、ISR、分区Leader
故障恢复             Quorum Queue 自动恢复         ISR 选举自动恢复
运维复杂度           中等                        较高
```

## 六、Spring 集成对比

```java
// RabbitMQ Spring Boot 配置
spring.rabbitmq.host=localhost
spring.rabbitmq.port=5672
spring.rabbitmq.username=guest
spring.rabbitmq.password=guest
spring.rabbitmq.listener.simple.acknowledge-mode=manual
spring.rabbitmq.listener.simple.prefetch=10

// Kafka Spring Boot 配置
spring.kafka.bootstrap-servers=localhost:9092
spring.kafka.consumer.group-id=my-group
spring.kafka.consumer.auto-offset-reset=earliest
spring.kafka.consumer.enable-auto-commit=false
spring.kafka.listener.ack-mode=MANUAL_IMMEDIATE
spring.kafka.producer.acks=all
spring.kafka.producer.retries=3
```

## 七、选型建议

1. **业务消息选 RabbitMQ**，路由灵活、延迟低
2. **大数据/日志选 Kafka**，吞吐高、可回溯
3. **金融/电商可选 RocketMQ**，事务消息、延迟消息
4. **小团队直接用 RabbitMQ**，学习成本最低
5. **不要追求一个 MQ 解决所有问题**，混合使用是常见做法
6. **已有 Java 技术栈优先考虑 Kafka 或 RocketMQ**，运维生态更丰富
7. **需要复杂路由和消息确认选 RabbitMQ**，AMQP 协议原生支持
