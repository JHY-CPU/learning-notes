# RocketMQ 概述

## 一、什么是 RocketMQ

Apache RocketMQ 是阿里巴巴开源的**分布式消息中间件**，使用 Java 开发。以高可靠、高吞吐、低延迟著称，原生支持事务消息、延迟消息等高级特性。

```
核心特点:
├── 事务消息 - 原生支持分布式事务
├── 延迟消息 - 18 个延迟级别
├── 消息回溯 - 按时间/Offset 回溯
├── 消息过滤 - Tag/SQL92 表达式
├── 消息轨迹 - 全链路追踪
└── 万亿级吞吐 - 阿里双 11 验证
```

## 二、架构概览

```
Producer                  NameServer
   │                         │
   ├── 注册/发现 ──────────→ │
   │                         │
   ├── 发送消息 ──→ Broker (Master)
   │                   │
   │              Broker (Slave)
   │
Consumer
   │
   ├── 订阅消息 ←── Broker 推送
```

## 三、与 Kafka 对比

| 特性 | RocketMQ | Kafka |
|------|----------|-------|
| 事务消息 | 原生支持 | 0.11+ 支持 |
| 延迟消息 | 18 级别 | 不支持 |
| 消息过滤 | Tag/SQL92 | 不支持 |
| 消息回溯 | 按时间 | 按 Offset |
| 消息轨迹 | 内置 | 需额外实现 |
| 吞吐量 | 十万级 | 百万级 |
| 开发语言 | Java | Java/Scala |

## 四、适用场景

```java
// 1. 电商交易 - 事务消息
rocketMQTemplate.sendMessageInTransaction("order-topic", msg, order);

// 2. 订单超时 - 延迟消息
rocketMQTemplate.syncSend("delay-topic", msg, 3000, 16); // 30分钟

// 3. 金融业务 - 高可靠性
// 同步刷盘 + 同步复制
```

## 五、版本演进

```yaml
版本历史:
  3.x: 阿里内部版本
  4.x: Apache 开源版本
  4.5+: 事务消息优化
  4.8+: DLedger 模式
  4.9+: 消息轨迹增强
  5.0: 新一代架构，gRPC 协议
```

## 六、注意事项

1. **RocketMQ 在国内使用广泛**，中文文档和社区支持好
2. **事务消息是 RocketMQ 的核心优势**，电商场景首选
3. **5.0 版本不完全向下兼容**，升级需谨慎
4. **NameServer 是无状态的**，可随意扩展
5. **消费失败重试策略内置支持**，不需要额外实现
