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

## 二、注意事项

1. **TCC 适合资金类场景**
2. **需要实现三个接口**
3. **Try 阶段要预留资源**
4. **Confirm 和 Cancel 要幂等**
5. **空回滚和悬挂要处理**
