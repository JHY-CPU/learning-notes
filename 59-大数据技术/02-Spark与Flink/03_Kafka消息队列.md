# Kafka消息队列


## Kafka消息队列


大数据Kafka消息队列


Apache Kafka 是一个分布式流处理平台，广泛用于构建实时数据管道和流式应用。


## Kafka核心架构


Kafka 采用发布-订阅模式，基于分布式提交日志（Commit Log）实现高吞吐、低延迟的消息传递。


```
┌────────────────────────────────────────────────────────────┐
│                    Kafka 架构                               │
│                                                            │
│  Producer (生产者)                                          │
│  ├── 将消息发送到指定的 Topic                                │
│  ├── 支持同步/异步发送                                       │
│  └── 通过分区器(Partitioner)决定消息写入哪个分区              │
│                                                            │
│  Broker (代理服务器)                                        │
│  ├── Kafka 集群中的单个节点                                  │
│  ├── 每个 Broker 可管理多个分区                               │
│  └── 通过 ZooKeeper/KRaft 进行集群协调                       │
│                                                            │
│  Consumer (消费者)                                          │
│  ├── 从 Topic 中拉取消息                                    │
│  ├── 属于某个 Consumer Group                                │
│  └── 每个分区仅被同一 Group 中的一个消费者消费                 │
│                                                            │
│  Topic (主题)                                               │
│  ├── 消息的逻辑分类                                         │
│  ├── 每个 Topic 可包含多个 Partition                         │
│  └── 消息按时间顺序追加，不可修改                             │
└────────────────────────────────────────────────────────────┘
```


> **Note:** Kafka 3.x 开始使用 KRaft 协议替代 ZooKeeper，简化了部署和运维。


## 分区（Partition）机制


分区是 Kafka 实现并行处理和水平扩展的核心单元。


```
分区特性：
- 每个分区是一个有序的、不可变的消息序列
- 每条消息有一个唯一的 offset（偏移量）
- 分区内的消息严格有序，跨分区不保证全局有序
- 分区策略：
  * 指定 Key：hash(key) % numPartitions
  * 不指定 Key：轮询或粘性分区（Sticky Partitioner）
  * 自定义 Partitioner：实现 Partitioner 接口

副本机制（Replication）：
┌──────────────────────────────────────┐
│  Partition-0                         │
│  ├── Leader Replica (读写)           │
│  ├── Follower Replica 1 (同步备份)   │
│  └── Follower Replica 2 (同步备份)   │
│                                      │
│  ISR (In-Sync Replicas)：            │
│  与 Leader 保持同步的副本集合          │
│  Leader 选举仅从 ISR 中选取           │
└──────────────────────────────────────┘
```


## 消费者组（Consumer Group）


消费者组是 Kafka 实现消息广播和负载均衡的关键概念。


```
消费者组工作原理：
┌──────────────────────────────────────────────────────┐
│  Topic (4个分区)                                      │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐                        │
│  │ P0 │ │ P1 │ │ P2 │ │ P3 │                        │
│  └─┬──┘ └─┬──┘ └─┬──┘ └─┬──┘                        │
│    │      │      │      │                            │
│    ▼      ▼      ▼      ▼   Consumer Group A         │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐                        │
│  │ C0 │ │ C1 │ │ C2 │ │ C3 │  (4个消费者，1对1消费)  │
│  └────┘ └────┘ └────┘ └────┘                        │
│                                                      │
│  同一 Group 内：消息负载均衡（每个分区只被一个消费者消费）│
│  不同 Group 之间：消息广播（每组都能收到全部消息）       │
└──────────────────────────────────────────────────────┘

消费者重平衡（Rebalance）：
- 触发条件：消费者加入/离开、分区数变更
- 重平衡策略：Range、RoundRobin、Sticky
- Rebalance 期间消费会暂停，应尽量减少触发
```


## Exactly-Once 语义


Kafka 提供端到端的精确一次（Exactly-Once）语义保证。


```
消息投递语义对比：
┌─────────────┬───────────────────────────────────────┐
│ At-Most-Once │ 消息可能丢失，但不会重复                  │
│              │ 实现：不等确认就提交 offset               │
├─────────────┼───────────────────────────────────────┤
│ At-Least-Once│ 消息不丢失，但可能重复                    │
│              │ 实现：处理完再提交 offset（可能重复消费）   │
├─────────────┼───────────────────────────────────────┤
│ Exactly-Once │ 消息不丢失且不重复                       │
│              │ 实现：事务 + 幂等生产者                   │
└─────────────┴───────────────────────────────────────┘

Exactly-Once 实现机制：
1. 幂等生产者（Idempotent Producer）
   - 设置 enable.idempotence=true
   - Producer 为每条消息分配 Sequence Number
   - Broker 根据 (ProducerID, Partition, Sequence) 去重
   - 仅保证单分区、单会话的幂等性

2. 事务（Transactions）
   - 跨分区的原子写入
   - Transaction Coordinator 管理事务状态
   - 支持：consume-transform-produce 流程
   - read_committed / read_uncommitted 隔离级别
```


> **Note:** 幂等性和事务会带来一定的性能开销，应根据业务场景选择合适的投递语义。


## 日志压缩（Log Compaction）


日志压缩保留每个 Key 的最新值，适用于状态存储和变更数据捕获（CDC）场景。


```
日志压缩 vs 日志删除：
┌────────────────────────────────────────────────────────┐
│ 日志删除（Delete）：                                     │
│   按时间或大小删除旧消息（默认策略）                        │
│   适用于：事件流、日志收集                                │
│                                                        │
│ 日志压缩（Compact）：                                    │
│   保留每个 Key 的最新 Value                              │
│   清理后台线程定期合并相同 Key 的记录                       │
│   适用于：配置变更、用户状态、CDC                          │
└────────────────────────────────────────────────────────┘

日志压缩工作原理：
offset: 0    1    2    3    4    5    6    7
key:    K1   K2   K1   K3   K2   K1   K3   K1
value:  V1   V2   V3   V4   V5   V6   V7   V8

压缩后（保留每个Key最新值）：
offset:              5    7    6
key:                 K1   K1   K3
value:               V6   V8   V7

配置参数：
- cleanup.policy=compact
- min.compaction.lag.ms：消息保留的最小时间
- delete.retention.ms：墓碑标记的保留时间
```


> **Note:** 日志压缩保证同一 Key 的消息在同一分区内，因此 Key 的选择至关重要。


<!-- Converted from: 03_Kafka消息队列.html -->
