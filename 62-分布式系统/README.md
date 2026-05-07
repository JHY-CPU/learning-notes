# 60-分布式系统

> 分布式系统是现代互联网架构的基石，涵盖从一致性理论到工程实践的完整知识体系。

---

## 目录索引

| 序号 | 主题 | 说明 |
|------|------|------|
| [01](01-一致性理论/) | 一致性理论 | CAP/BASE、一致性模型、时钟与事件序、拜占庭问题、FLP不可能性 |
| [02](02-共识算法/) | 共识算法 | Paxos、Raft、ZAB、Viewstamped Replication 及工程选型 |
| [03](03-分布式事务/) | 分布式事务 | 2PC/3PC、TCC、Saga、事务性发件箱、Seata框架 |
| [04](04-分布式存储与计算/) | 分布式存储与计算 | GFS/HDFS、Dynamo/Spanner、MapReduce/Spark/Flink、Kafka |

---

## 学习路线建议

```
一致性理论基础 → 共识算法原理 → 分布式事务模式 → 存储与计算系统
```

**第一阶段：理论基础** — 理解CAP、BASE、一致性模型等核心概念，建立分布式系统的思维方式。

**第二阶段：算法内核** — 深入Paxos、Raft、ZAB等共识算法，理解分布式系统如何在节点故障下达成一致。

**第三阶段：事务模式** — 掌握2PC、3PC、Saga、TCC等事务协调模式，学会在一致性与可用性之间权衡。

**第四阶段：系统架构** — 了解GFS、Spanner、Kafka等经典分布式系统的架构设计，掌握实际工程中的分布式解决方案。

---

## 推荐阅读

- Martin Kleppmann, 《数据密集型应用系统设计》(DDIA)
- Andrew S. Tanenbaum, 《分布式系统原理与范型》
- Leslie Lamport, Paxos Made Simple
- Diego Ongaro, Raft论文 In Search of an Understandable Consensus Algorithm
- James C. Corbett et al., Google Spanner论文
