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

## 三、Pulsar 订阅模式

```java
// 独占订阅（单消费者）
Consumer<String> exclusiveConsumer = client.newConsumer(Schema.STRING)
    .topic("orders")
    .subscriptionName("order-group")
    .subscriptionType(SubscriptionType.Exclusive)
    .subscribe();

// 共享订阅（多消费者负载均衡）
Consumer<String> sharedConsumer = client.newConsumer(Schema.STRING)
    .topic("orders")
    .subscriptionName("order-group")
    .subscriptionType(SubscriptionType.Shared)
    .subscribe();

// Failover 订阅（主备模式）
Consumer<String> failoverConsumer = client.newConsumer(Schema.STRING)
    .topic("orders")
    .subscriptionName("order-group")
    .subscriptionType(SubscriptionType.Failover)
    .subscribe();

// Key_Shared 订阅（相同 Key 到同一消费者）
Consumer<String> keySharedConsumer = client.newConsumer(Schema.STRING)
    .topic("orders")
    .subscriptionName("order-group")
    .subscriptionType(SubscriptionType.Key_Shared)
    .subscribe();
```

## 四、Pulsar Functions

```java
// 轻量级流处理函数
public class OrderEnrichment implements Function<String, String> {
    @Override
    public String process(String input, Context context) {
        Order order = JSON.parseObject(input, Order.class);
        // 查询用户信息
        User user = userService.getUser(order.getUserId());
        order.setUserName(user.getName());
        return JSON.toJSONString(order);
    }
}

// 部署 Function
// pulsar-admin functions create \
//   --jar order-enrichment.jar \
//   --classname OrderEnrichment \
//   --inputs raw-orders \
//   --output enriched-orders
```

## 五、Pulsar vs Kafka 架构对比

```
维度              Pulsar                         Kafka
存储              BookKeeper (独立存储层)          Broker 内置存储
计算              Broker (无状态)                 Broker (有状态)
扩缩容            Broker 秒级扩展                 需要 Rebalance
多租户            原生支持                        需外部方案
地域复制          内置 Geo-Replication            MirrorMaker 2
队列+流           原生统一模型                     仅流模型
运维复杂度        高 (BookKeeper)                 中等
```

## 六、注意事项

1. **Pulsar 存算分离适合云原生部署**
2. **多租户是核心优势**
3. **运维复杂度高于 Kafka**
4. **社区生态在快速发展中**
5. **适合需要多租户隔离的企业级场景**
6. **Pulsar Functions 适合轻量 ETL**，重计算用 Flink
7. **BookKeeper 的磁盘 IO 是性能瓶颈**，建议使用 SSD
