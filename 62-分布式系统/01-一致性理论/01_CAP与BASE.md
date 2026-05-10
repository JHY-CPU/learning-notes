# CAP与BASE


## CAP与BASE


分布式CAP一致性


CAP定理和BASE理论是分布式系统设计的基石理论。


## CAP定理


```
CAP定理（Brewer's Conjecture, 2000年）：
分布式系统不可能同时满足以下三个特性中的全部：

┌─────────────────────────────────────────────┐
│  C - Consistency（一致性）                    │
│  所有节点在同一时间看到相同的数据              │
│  每次读取都能获得最近写入的数据                │
│                                             │
│  A - Availability（可用性）                   │
│  每个请求都能在合理时间内获得响应              │
│  不保证返回最新数据，但保证不报错              │
│                                             │
│  P - Partition Tolerance（分区容错性）         │
│  网络分区时系统继续运行                       │
│  节点间通信失败时系统仍然可用                  │
└─────────────────────────────────────────────┘

关键认识：P是必须选择的！
- 分布式系统必然存在网络分区
- 实际选择是：CP 还是 AP

CP系统（选择一致性）：
- 网络分区时拒绝部分请求，保证一致性
- 例：ZooKeeper, etcd, HBase, Redis Cluster
- 适用：金融交易、库存管理

AP系统（选择可用性）：
- 网络分区时继续服务，可能返回旧数据
- 例：Cassandra, DynamoDB, Eureka, CouchDB
- 适用：社交网络、推荐系统、DNS
```


## PACELC扩展


```
PACELC（对CAP的扩展，2012年）：

If Partition → choose A or C
Else → choose Latency or Consistency

┌─────────────────────────────────────────────┐
│  分区发生时：选择A或C（同CAP）                │
│  正常运行时：选择延迟(L)或一致性(C)           │
└─────────────────────────────────────────────┘

系统分类：
- PA/EL：分区时选A，正常时选低延迟
  → Cassandra, DynamoDB, CouchDB
- PA/EC：分区时选A，正常时选一致性
  → Eureka
- PC/EL：分区时选C，正常时选低延迟
  → MongoDB, Riak
- PC/EC：分区时选C，正常时选一致性
  → ZooKeeper, HBase, Redis

更细粒度地描述了分布式系统的一致性权衡。
```


## BASE理论


```
BASE（Basically Available, Soft State, Eventually Consistent）
是对AP系统的总结：

BA - Basically Available（基本可用）：
- 系统在部分故障时仍能提供核心功能
- 响应时间可能延长（降级）
- 功能可能缩减（熔断）

SS - Soft State（软状态）：
- 允许系统中的数据存在中间状态
- 不要求实时强一致
- 数据的不一致是可接受的

EC - Eventually Consistent（最终一致）：
- 在没有新更新的情况下，最终所有副本一致
- 不保证何时达到一致（通常是毫秒到秒级）
- 一致性模型：因果一致、会话一致、读己之所写

ACID vs BASE：
┌──────────┬────────────────┬────────────────┐
│          │ ACID           │ BASE           │
├──────────┼────────────────┼────────────────┤
│ 理念      │ 强一致         │ 最终一致        │
│ 场景      │ 关系型数据库   │ NoSQL/分布式    │
│ 可用性    │ 低（可能阻塞） │ 高              │
│ 一致性    │ 强             │ 最终            │
│ 典型系统  │ MySQL/Postgres │ Cassandra/Redis│
└──────────┴────────────────┴────────────────┘
```


> **Note:** 现代分布式系统往往混合使用：核心数据用ACID，非核心数据用BASE，通过业务逻辑保证最终一致性。


<!-- Converted from: 01_CAP与BASE.html -->
