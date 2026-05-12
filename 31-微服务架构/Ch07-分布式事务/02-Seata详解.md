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

## 三、工作原理

Seata AT 模式基于两阶段提交改进实现。第一阶段（Branch Commit 前）：TM 开启全局事务，RM 在本地事务执行业务 SQL 的同时，生成 undo_log（前镜像和后镜像）并提交本地事务，由 TC 记录全局事务状态。第二阶段（全局提交/回滚）：若所有分支成功，TC 异步删除 undo_log；若任一分支失败，TC 通知各分支回滚，RM 根据 undo_log 反向恢复数据。TC 是独立服务，TM 和 RM 以 SDK 形式嵌入业务应用。

## 四、优缺点

**优点：**
- AT 模式对业务代码零侵入，仅需 @GlobalTransactional 注解
- 支持 AT、TCC、Saga、XA 四种模式，覆盖多种场景
- 国内社区活跃，Spring Cloud Alibaba 深度集成

**缺点：**
- AT 模式依赖 undo_log 表，增加数据库存储
- 全局锁在高并发场景下可能成为瓶颈
- TC 服务需要高可用部署，增加运维成本

## 五、最佳实践

1. TC 部署至少 3 节点集群，使用 DB 存储模式保障高可用
2. 全局事务超时设置合理（通常 30-60 秒），避免资源长期锁定
3. undo_log 表定期清理，防止存储膨胀
4. 高并发场景考虑 TCC 模式替代 AT，避免全局锁

## 六、常见陷阱

1. **undo_log 表未创建或字段不正确**，分支事务注册失败
2. **全局事务超时过长**，资源锁定时间过长影响吞吐
3. **TC 服务单机部署**，宕机后全局事务无法协调
4. **AT 模式下部分 SQL 不支持**（如不支持 DDL），需要手动处理
