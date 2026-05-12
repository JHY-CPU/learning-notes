# 事件驱动与CQRS详解


## 事件驱动架构与CQRS详解

一、事件驱动架构

## 一、事件驱动架构 (Event-Driven Architecture)


事件驱动架构以事件的产生、检测和消费为核心，将系统组件通过异步事件松耦合地连接在一起。


### 1.1 核心概念


| 概念 | 定义 | 示例 |
| --- | --- | --- |
| 事件 (Event) | 系统中发生的事实记录，不可变 | OrderCreated, PaymentCompleted |
| 事件生产者 | 检测到状态变化并发布事件的组件 | 订单服务发布OrderCreated |
| 事件消费者 | 订阅并响应事件的组件 | 库存服务订阅OrderCreated |
| 事件通道 | 事件传输的中间件 | Kafka, RabbitMQ, Pulsar |
| 事件存储 (Event Store) | 持久化所有事件的日志 | EventStoreDB, Kafka (日志保留) |


### 1.2 消息中间件对比


| 中间件 | 模型 | 特点 | 适用场景 |
| --- | --- | --- | --- |
| Kafka | 发布/订阅 + 分区 | 高吞吐、持久化、可回溯 | 事件流、日志聚合、大数据管道 |
| RabbitMQ | 队列 + 路由 | 灵活路由、可靠投递 | 任务分发、RPC、复杂路由 |
| Pulsar | 发布/订阅 + 分层 | 存算分离、多租户 | 云原生、多租户场景 |
| Redis Streams | 日志 + 消费者组 | 轻量、内存级速度 | 轻量级事件流、实时通知 |


### 1.3 事件设计原则


- **事件不可变：**
   事件代表已发生的事实，不可修改或删除
- **事件自包含：**
   事件应包含足够的上下文信息，消费者不需回查
- **事件命名过去式：**
   `OrderCreated`
   而非
   `CreateOrder`
- **事件版本管理：**
   使用Schema Registry管理事件结构演化
- **幂等消费：**
   消费者必须处理重复事件（At-least-once投递）

二、Event Sourcing

## 二、Event Sourcing（事件溯源）


Event Sourcing 不存储实体的当前状态，而是存储导致状态变化的所有事件序列。当前状态通过回放事件重建。


### 2.1 传统 vs Event Sourcing


| 维度 | 传统CRUD | Event Sourcing |
| --- | --- | --- |
| 存储内容 | 实体当前状态 | 状态变更事件序列 |
| 更新操作 | UPDATE覆盖旧值 | APPEND新事件 |
| 历史追溯 | 需额外审计表 | 天然支持（事件即历史） |
| 状态重建 | 直接查询 | 回放事件序列 |
| 数据丢失 | UPDATE会丢失旧值 | 所有变更永久保留 |


### 2.2 Event Store 设计


> **Note:** **Event Store 数据模型：**
>
>
>
>
> events表：
>
>
> - stream_id: UUID (聚合根ID，如订单ID)
>
>
> - stream_type: string (如"Order")
>
>
> - version: bigint (事件序号，用于乐观并发控制)
>
>
> - event_type: string (如"OrderCreated")
>
>
> - event_data: JSONB (事件具体内容)
>
>
> - metadata: JSONB (时间戳、用户ID、因果ID等)
>
>
> - created_at: timestamp
>
>
>
>
> 关键约束：UNIQUE(stream_id, version) 保证事件顺序


### 2.3 快照 (Snapshot) 优化


当事件序列很长时，每次重建状态需要回放大量事件。快照机制定期保存聚合根的状态快照，重建时从最近的快照开始回放。


- 每N个事件创建一次快照（如每100个事件）
- 快照存储聚合根的完整状态
- 重建时：加载快照 + 回放快照之后的事件
- 快照是性能优化，可随时删除和重建

三、CQRS

## 三、CQRS (Command Query Responsibility Segregation)


CQRS 将系统的读操作（Query）和写操作（Command）分离到不同的模型和存储中，各自独立优化。


### 3.1 CQRS架构


| 组件 | 职责 | 特点 |
| --- | --- | --- |
| Command端（写模型） | 处理业务命令、验证、产生事件 | 强一致性、领域模型丰富 |
| Query端（读模型） | 响应查询请求 | 最终一致、视图优化、高性能 |
| 事件通道 | 将写端的事件传递给读端 | Kafka/RabbitMQ |
| 读模型投影 (Projection) | 将事件转换为查询优化的视图 | 可重建、可并行 |


### 3.2 读模型设计


> **Note:** **读模型示例（订单列表查询）：**
>
>
>
>
> 传统方式：JOIN orders + order_items + products + users（复杂查询）
>
>
>
>
> CQRS读模型：一个预计算的文档
>
>
> {
>
>
> "orderId": "ORD-001",
>
>
> "userName": "张三",
>
>
> "productName": "iPhone 15",
>
>
> "quantity": 2,
>
>
> "totalAmount": 13998,
>
>
> "status": "已支付",
>
>
> "createdAt": "2024-01-15T10:30:00Z"
>
>
> }
>
>
>
>
> 一次查询即可获得所有信息，无需JOIN


### 3.3 最终一致性


CQRS的读模型和写模型之间是最终一致的，这意味着写入后立即查询可能读不到最新数据。


- **可接受场景：**
   商品列表、评论列表、统计数据
- **不可接受场景：**
   余额查询、库存扣减（需要强一致）
- **处理策略：**
   UI层面显示"处理中"、写后延迟读、版本号轮询

四、Event Sourcing + CQRS

## 四、Event Sourcing + CQRS 组合


Event Sourcing和CQRS经常一起使用：Event Sourcing提供写模型的事件存储，CQRS提供读模型的查询优化视图。


### 4.1 完整架构流程

**数据流向：**


1. 用户发送Command → Command Handler


2. Command Handler加载聚合根（回放事件重建状态）


3. 执行业务逻辑 → 产生新事件


4. 新事件持久化到Event Store


5. 事件发布到消息通道


6. Projection Handler消费事件


7. 更新读模型（关系型DB/文档DB/搜索引擎）


8. 用户查询 → 从读模型返回预计算视图

### 4.2 适用场景与注意事项


| 适用场景 | 不适用场景 |
| --- | --- |
| 需要完整审计日志 | 简单CRUD应用 |
| 复杂领域逻辑 | 团队不熟悉DDD |
| 读写负载差异大 | 需要强一致性查询 |
| 事件驱动的微服务 | 小规模项目（过度设计） |
| 需要时间旅行（回溯任意时刻状态） | 对延迟极其敏感的场景 |


> **Note:** **复杂度警告：**
> Event Sourcing + CQRS 带来了事件版本管理、最终一致性处理、投影重建等额外复杂度。团队需要充分评估是否真正需要这些能力，避免过度设计。建议从简单CRUD开始，当遇到具体痛点时再引入。

========================================
  文件总结
========================================
  主题：事件驱动架构与CQRS详解
  内容概要：
    1. 事件驱动架构 - 事件/生产者/消费者/通道模型，Kafka vs RabbitMQ对比
    2. Event Sourcing - 存储事件序列而非当前状态，快照优化
    3. CQRS - 读写分离，读模型预计算优化查询
    4. ES+CQRS组合 - 完整架构流程和适用场景
  重点知识：
    - 事件不可变、自包含、命名过去式
    - Event Store的UNIQUE(stream_id, version)乐观并发控制
    - CQRS的最终一致性处理策略
    - ES+CQRS的复杂度警告，避免过度设计
========================================


<!-- Converted from: 04_事件驱动与CQRS详解.html -->
