# Apache Pulsar

## 一、Pulsar 简介

Apache Pulsar 是一个**云原生、多租户**的分布式消息流平台，存算分离架构。

```
架构特点:
├── 存算分离 - Broker 无状态，BookKeeper 存储
├── 多租户 - 原生支持命名空间隔离
├── 统一模型 - 队列 + 流
├── 跨地域复制 - 内置 Geo-Replication
└── Pulsar Functions - 轻量级流处理
```

## 二、基本使用

```java
// 生产者
PulsarClient client = PulsarClient.builder()
    .serviceUrl("pulsar://localhost:6650")
    .build();

Producer<String> producer = client.newProducer(Schema.STRING)
    .topic("my-topic")
    .create();
producer.send("Hello Pulsar");

// 消费者
Consumer<String> consumer = client.newConsumer(Schema.STRING)
    .topic("my-topic")
    .subscriptionName("my-sub")
    .subscribe();
Message<String> msg = consumer.receive();
```

## 三、注意事项

1. **Pulsar 存算分离适合云原生部署**
2. **多租户是核心优势**
3. **运维复杂度高于 Kafka**
4. **社区生态在快速发展中**
5. **适合需要多租户隔离的企业级场景**
