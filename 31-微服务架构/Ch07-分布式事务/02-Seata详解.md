# Seata 详解

## 一、Seata 架构

```
Seata 组件:
├── TC (Transaction Coordinator) - 事务协调者
├── TM (Transaction Manager) - 事务管理器
└── RM (Resource Manager) - 资源管理器
```

## 二、AT 模式

```java
// 全局事务
@GlobalTransactional
public void createOrder(Order order) {
    // 本地事务
    orderMapper.insert(order);
    // 远程调用
    inventoryClient.deduct(order);
    accountClient.deduct(order);
}
```

## 三、注意事项

1. **AT 模式是最常用的模式**
2. **需要 undo_log 表**
3. **Seata Server 需要高可用**
4. **全局事务有超时限制**
5. **性能开销比本地事务大**
