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

## 二、工作原理

Saga 将长事务拆分为一系列本地事务，每个本地事务有对应的补偿操作。正向流程按顺序执行各步骤，若某步骤失败，则按逆序执行已完成步骤的补偿操作。编排式 Saga 由一个中心协调器统一控制流程，适合步骤少的简单流程。协调式 Saga 各服务通过事件驱动，每个服务完成操作后发布事件触发下一步，适合复杂流程。Saga 不锁定资源，允许中间状态短暂不一致。

## 三、优缺点

**优点：**
- 无全局锁，吞吐量高
- 适合长时间运行的业务流程（如订单履约）
- 补偿逻辑清晰，可追溯

**缺点：**
- 仅保证最终一致性，中间状态对外可见
- 补偿操作可能失败，需要额外的恢复机制
- 编排式 Saga 协调器成为单点

## 四、最佳实践

1. 每个步骤的补偿操作必须幂等
2. 补偿操作失败时记录到死信队列，人工介入
3. 编排式使用状态机引擎（如 Seata Saga StateMachine）管理流程
4. 避免 Saga 步骤过多，建议不超过 10 步

## 五、Seata Saga 状态机

```json
{
  "Name": "orderSaga",
  "StartState": "CreateOrder",
  "States": {
    "CreateOrder": {
      "Type": "ServiceTask",
      "ServiceName": "orderService",
      "ServiceMethod": "create",
      "Next": "ReserveInventory",
      "CompensateState": "CancelOrder"
    },
    "ReserveInventory": {
      "Type": "ServiceTask",
      "ServiceName": "inventoryService",
      "ServiceMethod": "reserve",
      "Next": "DeductBalance",
      "CompensateState": "ReleaseInventory",
      "Retry": {
        "MaxAttempts": 3,
        "Backoff": "1s"
      }
    },
    "DeductBalance": {
      "Type": "ServiceTask",
      "ServiceName": "accountService",
      "ServiceMethod": "deduct",
      "Next": "Success",
      "CompensateState": "RefundBalance"
    },
    "Success": { "Type": "Succeed" },
    "CancelOrder": {
      "Type": "ServiceTask",
      "ServiceName": "orderService",
      "ServiceMethod": "cancel"
    },
    "ReleaseInventory": {
      "Type": "ServiceTask",
      "ServiceName": "inventoryService",
      "ServiceMethod": "release"
    },
    "RefundBalance": {
      "Type": "ServiceTask",
      "ServiceName": "accountService",
      "ServiceMethod": "refund"
    }
  }
}
```

## 六、Saga vs TCC 对比

| 维度 | Saga | TCC |
|------|------|-----|
| 一致性 | 最终一致 | 强一致 |
| 性能 | 高（无锁） | 中（资源冻结） |
| 复杂度 | 中（补偿逻辑） | 高（Try/Confirm/Cancel） |
| 资源锁定 | 不锁定 | Try 阶段冻结 |
| 适用场景 | 长事务、非资金类 | 资金类、强一致要求 |
| 中间状态 | 可见 | 不可见 |

## 七、常见陷阱

1. **补偿操作不幂等** - 重试导致数据异常，必须保证幂等
2. **补偿操作本身也会失败** - 需要重试或死信队列 + 人工介入
3. **Saga 中间状态被外部系统读取** - 导致业务逻辑错误
4. **编排式协调器宕机** - 进行中的 Saga 无人接管，需要持久化状态
5. **Saga 步骤过多** - 超过 10 步时协调复杂度急剧增加
