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

## 三、注意事项

1. **NATS 延迟极低**，适合实时性要求高的场景
2. **JetStream 提供持久化保证**
3. **NATS 集群部署简单**，内置集群支持
4. **社区和生态不如 Kafka/RabbitMQ**
5. **适合微服务间轻量通信**
