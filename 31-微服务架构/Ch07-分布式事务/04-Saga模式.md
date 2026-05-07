# Saga 模式

## 一、Saga 流程

```java
// 编排式 Saga
public class OrderSaga {
    // 正向流程
    public void execute(Order order) {
        inventoryService.reserve(order);  // Step 1
        accountService.deduct(order);     // Step 2
        orderService.create(order);       // Step 3
    }

    // 补偿流程
    public void compensate(Order order, int failedStep) {
        switch (failedStep) {
            case 3: accountService.refund(order);
            case 2: inventoryService.release(order);
        }
    }
}
```

## 二、注意事项

1. **Saga 适合长时间事务**
2. **补偿操作要幂等**
3. **编排式适合简单流程**
4. **协调式适合复杂流程**
5. **Seata 支持 Saga 模式**
