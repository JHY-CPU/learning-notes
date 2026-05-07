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

## 三、注意事项

1. **读写分离需要保证最终一致性**
2. **读库延迟要监控**，确保数据及时同步
3. **适合读多写少、读写比例差异大的场景**
4. **增加了系统复杂度**，不适合简单的 CRUD
5. **Elasticsearch 是常用的读库选择**
