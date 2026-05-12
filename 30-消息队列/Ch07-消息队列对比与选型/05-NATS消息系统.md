# NATS 消息系统

## 一、NATS 简介

NATS 是一个**云原生、高性能**的消息系统，Go 语言开发，延迟极低（微秒级）。

```go
// NATS 核心概念
// Subject: 消息主题（类似 Topic）
// Publisher: 发布者
// Subscriber: 订阅者
// JetStream: 持久化层
```

## 二、基本使用

```java
// 连接 NATS
Connection nc = Nats.connect("nats://localhost:4222");

// 发布
nc.publish("orders.created", orderJson.getBytes());

// 订阅
Subscription sub = nc.subscribe("orders.created");
Message msg = sub.nextMessage(Duration.ofMillis(1000));

// JetStream 持久化
JetStream js = nc.jetStream();
js.publish("orders", orderJson.getBytes());
```

## 三、JetStream 持久化详解

```java
// 创建持久化Stream
JetStreamManagement jsm = nc.jetStreamManagement();
StreamConfiguration sc = StreamConfiguration.builder()
    .name("ORDERS")
    .subjects("orders.>")
    .storage(StorageType.File)
    .retentionPolicy(RetentionPolicy.Limits)
    .maxMsgs(1000000)
    .build();
jsm.addStream(sc);

// 消费者拉取消息
JetStream js = nc.jetStream();
PullSubscribeOptions so = PullSubscribeOptions.builder()
    .durable("order-processor")
    .build();
JetStreamSubscription sub = js.subscribe("orders.>", so);
List<Message> messages = sub.fetch(100, Duration.ofSeconds(1));
```

## 四、NATS vs Kafka vs RabbitMQ

| 特性 | NATS | Kafka | RabbitMQ |
|------|------|-------|----------|
| 延迟 | 微秒级 | 毫秒级 | 毫秒级 |
| 吞吐量 | 高 | 极高 | 中等 |
| 持久化 | JetStream | 原生支持 | 原生支持 |
| 协议 | 自有协议 | 自有协议 | AMQP |
| 复杂度 | 低 | 高 | 中等 |

## 五、常见陷阱

1. **内存模式默认**：NATS Core默认不持久化，服务重启消息丢失，需明确使用JetStream
2. **JetStream成熟度**：JetStream相对较新，生产环境需充分测试，注意版本升级兼容性
3. **消息大小限制**：NATS默认消息大小为1MB，大消息需调整`max_payload`或改用对象存储
4. **集群脑裂**：网络分区时可能出现脑裂，需要合理配置集群节点数（奇数节点）
