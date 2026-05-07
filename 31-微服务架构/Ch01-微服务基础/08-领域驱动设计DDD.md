# 领域驱动设计 DDD

## 一、核心概念

```
DDD 核心:
├── 限界上下文 (Bounded Context) - 微服务边界
├── 聚合根 (Aggregate Root) - 事务边界
├── 实体 (Entity) - 有唯一标识
├── 值对象 (Value Object) - 无唯一标识
├── 领域事件 (Domain Event) - 状态变更通知
└── 领域服务 (Domain Service) - 跨聚合逻辑
```

```java
// 聚合根 - 订单
@Entity
public class Order extends AggregateRoot<OrderId> {
    private OrderId id;
    private CustomerId customerId;
    private List<OrderItem> items = new ArrayList<>();
    private OrderStatus status;

    // 业务方法
    public void pay() {
        if (status != OrderStatus.CREATED) {
            throw new IllegalStateException("订单状态不合法");
        }
        this.status = OrderStatus.PAID;
        registerEvent(new OrderPaidEvent(this.id));
    }

    public void addItem(ProductId productId, int quantity, BigDecimal price) {
        items.add(new OrderItem(productId, quantity, price));
    }
}

// 值对象 - 金额
@Value
public class Money {
    BigDecimal amount;
    Currency currency;
}

// 领域事件
public class OrderPaidEvent extends DomainEvent {
    private final OrderId orderId;
}
```

## 二、限界上下文映射

```
用户上下文 --→ 订单上下文 --→ 支付上下文
   (Customer)    (Order)       (Payment)
     │               │              │
  用户库           订单库          支付库
```

## 三、DDD 与微服务

| DDD 概念 | 微服务对应 |
|---------|-----------|
| 限界上下文 | 微服务边界 |
| 聚合根 | 事务一致性边界 |
| 领域事件 | MQ 事件 |
| 上下文映射 | 服务间 API |

## 四、注意事项

1. **限界上下文是微服务拆分的核心依据**
2. **聚合根是事务一致性边界**，不要跨聚合根做事务
3. **领域事件驱动服务间通信**
4. **通用语言（Ubiquitous Language）统一团队术语**
5. **不要过度建模**，从简单开始，逐步演进
