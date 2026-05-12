# TCC 模式

## 一、TCC 三个阶段

```java
// Try - 预留资源
public boolean tryDeduct(Long orderId, Long productId, int quantity) {
    // 冻结库存
    return inventoryMapper.freeze(productId, quantity) > 0;
}

// Confirm - 确认提交
public void confirmDeduct(Long orderId) {
    // 扣减冻结库存
    inventoryMapper.confirmDeduct(orderId);
}

// Cancel - 取消释放
public void cancelDeduct(Long orderId) {
    // 释放冻结库存
    inventoryMapper.releaseFreeze(orderId);
}
```

## 二、工作原理

TCC（Try-Confirm-Cancel）是一种业务层面的两阶段提交。Try 阶段检查业务约束并预留资源（如冻结库存、冻结余额），不执行真正的业务操作。Confirm 阶段确认提交，将预留资源转为实际扣减。Cancel 阶段回滚，释放预留资源。三个阶段都需要实现为独立接口。TCC 必须处理三个异常场景：空回滚（Try 未执行但收到 Cancel）、幂等（Confirm/Cancel 可能被重复调用）、悬挂（Cancel 先于 Try 到达）。

## 三、优缺点

**优点：**
- 不依赖全局锁，性能优于 Seata AT 模式
- 资源预留机制精确控制并发冲突
- 适合资金、库存等对一致性要求极高的场景

**缺点：**
- 每个服务需实现 Try/Confirm/Cancel 三个接口，开发成本高
- 业务侵入性强，需要从业务层面设计资源预留逻辑
- 空回滚、幂等、悬挂三个问题处理复杂

## 四、最佳实践

1. 使用事务状态表记录 Try 执行状态，用于判断空回滚和幂等
2. Confirm 和 Cancel 方法必须实现幂等，根据事务状态表判断
3. Try 阶段使用业务乐观锁而非数据库行锁，减少锁冲突
4. 预留资源设置过期时间，防止长时间未 Confirm 或 Cancel

## 五、常见陷阱

1. **空回滚未处理**，Cancel 执行时找不到预留资源导致异常
2. **Confirm/Cancel 不幂等**，网络重试导致重复扣减
3. **悬挂问题**，Cancel 先到达释放资源后 Try 再到达又预留，导致资源无法回收
4. **Try 阶段做了真正的业务操作**，违背 TCC 设计原则
