# CQRS 模式

## 一、CQRS 概述

CQRS（Command Query Responsibility Segregation）将**读操作和写操作分离**，写入用命令模型，读取用查询模型。

```
Write Path:  Command -> MQ -> Write Service -> Write DB
Read Path:   Query -> Read Service -> Read DB (从 Write DB 同步)
```

## 二、MQ 实现 CQRS

```java
// 写操作 - 发布事件
@Transactional
public void createOrder(CreateOrderCommand cmd) {
    Order order = new Order(cmd);
    writeOrderMapper.insert(order);
    // 发布事件到 MQ
    kafkaTemplate.send("order-events", order.getId().toString(), JSON.toJSONString(order));
}

// 读模型消费者 - 同步到读库
@Component
@KafkaListener(topics = "order-events")
public class OrderReadModelConsumer {
    public void onEvent(ConsumerRecord<String, String> record) {
        Order order = JSON.parseObject(record.value(), Order.class);
        // 更新 Elasticsearch 读库
        elasticsearchTemplate.save(order);
    }
}

// 读操作 - 从读库查询
public Order getOrder(Long orderId) {
    return elasticsearchTemplate.get(orderId, Order.class);
}
```

## 三、读模型更新策略

```java
// 读模型消费者：更新多个读库
@Component
@KafkaListener(topics = "order-events", groupId = "read-model-updater")
public class ReadModelUpdater {
    // 更新 Elasticsearch
    @KafkaHandler
    public void updateES(ConsumerRecord<String, String> record) {
        OrderEvent event = JSON.parseObject(record.value(), OrderEvent.class);
        switch (event.getType()) {
            case "CREATED":
            case "UPDATED":
                elasticsearchTemplate.save(toOrderView(event));
                break;
            case "DELETED":
                elasticsearchTemplate.delete(event.getOrderId().toString(), OrderView.class);
                break;
        }
    }

    // 更新 Redis 缓存
    @KafkaHandler
    public void updateCache(ConsumerRecord<String, String> record) {
        OrderEvent event = JSON.parseObject(record.value(), OrderEvent.class);
        if ("CREATED".equals(event.getType()) || "UPDATED".equals(event.getType())) {
            redisTemplate.opsForValue().set(
                "order:" + event.getOrderId(),
                JSON.toJSONString(toOrderView(event)),
                30, TimeUnit.MINUTES
            );
        }
    }
}
```

## 四、多读模型架构

```
                    CQRS 多读模型
    ┌────────────────────────────────────────────────┐
    │                                                 │
    │   写库 (MySQL/PostgreSQL)                        │
    │      │                                          │
    │      ↓                                          │
    │   消息队列 (Kafka Topic)                          │
    │      │                                          │
    │      ├──→ ES 消费者 → Elasticsearch (全文搜索)     │
    │      ├──→ 缓存消费者 → Redis (热点查询)             │
    │      ├──→ 报表消费者 → ClickHouse (数据分析)        │
    │      └──→ 推送消费者 → WebSocket (实时通知)         │
    │                                                 │
    └────────────────────────────────────────────────┘
```

## 五、CQRS 与最终一致性

```java
// 检查读模型同步延迟
@Component
public class ReadModelLagMonitor {
    @Scheduled(fixedRate = 30000)
    public void checkLag() {
        // 写库最新版本
        long writeVersion = writeOrderMapper.getLatestVersion();
        // 读库最新版本
        long readVersion = elasticsearchTemplate.getLatestVersion();

        long lag = writeVersion - readVersion;
        if (lag > 1000) {
            alertService.send("读模型同步延迟告警", "延迟: " + lag + " 条");
        }
    }
}
```

## 六、注意事项

1. **读写分离需要保证最终一致性**
2. **读库延迟要监控**，确保数据及时同步
3. **适合读多写少、读写比例差异大的场景**
4. **增加了系统复杂度**，不适合简单的 CRUD
5. **Elasticsearch 是常用的读库选择**
6. **读模型要支持重放**，从事件流重新构建
7. **多读模型更新时考虑并行消费**，降低延迟
