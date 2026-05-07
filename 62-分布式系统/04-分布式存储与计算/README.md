# 04-分布式存储与计算

> 分布式存储与计算是大数据时代的技术基石。从 Google 的 GFS/MapReduce 到现代的 Kafka/Flink，本章系统梳理分布式存储与计算的核心架构和关键技术。

---

## 1. 分布式文件系统

### GFS（Google File System）

GFS 由 Google 于 2003 年发表论文，是现代分布式文件系统的鼻祖。

#### 架构组件

- **Master 节点**：管理文件系统的元数据（命名空间、文件到 Chunk 的映射、Chunk 位置信息），单点设计但元数据常驻内存。
- **Chunk Server**：存储实际的文件数据，文件被切分为固定大小的 Chunk（默认 64MB），每个 Chunk 有多个副本（默认 3 副本）。
- **Client**：向 Master 获取元数据，直接与 Chunk Server 进行数据读写。

#### 关键设计

- **大 Chunk 尺寸（64MB）**：减少元数据量，降低 Client 与 Master 的交互频率。
- **单 Master**：简化设计，Master 负责元数据，数据流不经过 Master。
- **租约机制（Lease）**：Master 将 Chunk 的写权限以租约形式授予一个 Chunk Server（Primary），由 Primary 决定写入顺序。
- **流式写入**：数据沿 Chunk Server 链流水线传输，最大化带宽。
- **弱一致性模型**：提供记录追加（Record Append）的原子性保证，但不保证所有副本完全一致。

#### 设计哲学

GFS 针对 Google 的工作负载做了大量优化：大文件、追加写多于随机写、高吞吐量优先于低延迟。

### HDFS（Hadoop Distributed File System）

HDFS 是 GFS 的开源实现，是 Hadoop 生态的核心存储层。

#### 与 GFS 的区别

| 特性 | GFS | HDFS |
|------|-----|------|
| 元数据管理 | 单 Master | NameNode（Active/Standby） |
| 数据块大小 | 64MB | 128MB（可配置） |
| 容错 | 无 HA（论文中） | 支持 NameNode HA（QJM） |
| 文件追加 | 支持 Record Append | 支持但受限 |
| 生态 | Google 内部使用 | 开源，与 MapReduce/Spark 深度集成 |

#### HDFS HA 架构

- 使用 QJM（Quorum Journal Manager）实现 NameNode 的 Active/Standby 高可用
- DataNode 向两个 NameNode 都发送心跳和块报告
- ZooKeeper 用于故障检测和自动切换

---

## 2. 分布式键值存储

### Amazon Dynamo

Dynamo 是 Amazon 于 2007 年发表论文的分布式键值存储，是 AP 系统的经典代表。

#### 核心技术

- **一致性哈希**：节点和数据映射到同一个环上，支持动态增删节点。
- **向量时钟**：检测并解决数据冲突，允许客户端处理多版本。
- **Quorum 机制**：NWR 模型，W + R > N 保证一致性语义。
- **Gossip 协议**：节点间通过 Gossip 协议传播成员信息和故障检测。
- **Sloppy Quorum 与 Hinted Handoff**：当目标节点不可用时，临时写入其他节点，待恢复后转发。
- **Merkle Tree**：用于副本间的数据同步和一致性校验。

### Apache Cassandra

Cassandra 由 Facebook 开源，融合了 Dynamo 的架构和 Google Bigtable 的数据模型。

#### 特点

- 无主架构（Masterless），所有节点对等
- 一致性级别可配置（ONE、QUORUM、ALL）
- 支持 CQL（类似 SQL 的查询语言）
- LSM-Tree 存储引擎，写入性能优异
- 适合写多读少、高可用场景

### etcd

etcd 是 Kubernetes 的核心存储，基于 Raft 的强一致键值存储。

#### 特点

- 强一致性（线性一致性读写）
- 基于 Raft 共识算法
- 支持 Watch 机制（高效事件通知）
- 使用 BoltDB（B+Tree）作为底层存储引擎
- 适合存储元数据和配置信息

---

## 3. 分布式数据库

### 分片策略

分片（Sharding）是将数据分散存储到多个节点上的技术。

#### 范围分片（Range Sharding）

- 按照键的范围将数据划分到不同节点
- **优点**：范围查询高效，数据局部性好
- **缺点**：可能出现热点（顺序写集中在最后一个分片）
- **适用场景**：时间序列数据、需要范围扫描的应用

#### 哈希分片（Hash Sharding）

- 对键取哈希值，按哈希值范围分配到节点
- **优点**：数据分布均匀，避免热点
- **缺点**：范围查询需要扫描所有分片，增加网络开销
- **适用场景**：点查询为主、数据分布不均匀的场景

#### 一致性哈希（Consistent Hashing）

- 将哈希空间组织为环，节点和数据映射到环上
- 数据由顺时针方向最近的节点负责
- **优点**：增删节点只影响相邻数据，迁移量最小
- **虚拟节点**：每个物理节点映射多个虚拟节点，改善负载均衡

### 数据复制

#### 主从复制（Leader-Follower / Master-Slave）

- 一个 Leader 处理写请求，多个 Follower 异步或同步复制数据
- 读请求可以分摊到 Follower
- **异步复制**：延迟低但可能丢数据
- **同步复制**：数据安全但延迟高
- **半同步**：多数派同步，兼顾安全与延迟

#### 多主复制（Multi-Leader）

- 多个节点都可以接受写请求
- 需要解决写冲突（通过向量时钟、Last-Write-Wins 或应用层解决）
- 适用场景：多数据中心部署、离线编辑

#### 无主复制（Leaderless）

- 所有节点对等，都可以接受读写请求
- 使用 Quorum 机制保证一致性
- 读修复（Read Repair）和反熵（Anti-Entropy）保证副本收敛
- 典型系统：Dynamo、Cassandra

### Google Spanner

Google Spanner 是全球分布式数据库的里程碑，首次在大规模系统中实现了外部一致性（线性一致性）。

#### 核心技术

- **TrueTime API**：通过原子钟和 GPS 提供有界时间不确定性（通常 < 7ms），使得事务可以基于时间戳排序
- **Paxos 分组**：数据按目录（Directory）组织，每个 Paxos 组管理一组目录，跨 Paxos 组使用两阶段提交
- **两阶段提交**：分布式事务通过 2PC 协调多个 Paxos 组
- **外部一致性保证**：事务的时间戳单调递增，且事务提交后的时间戳大于事务执行期间 TrueTime 的下界

#### 架构特点

```
客户端 → Spanner Proxy → Paxos Leader → 多个 Paxos Follower
                               ↓
                          TrueTime API（原子钟 + GPS）
```

- 全球分布的数据中心部署
- 自动分片和负载均衡
- 支持 SQL 查询（Spanner SQL）

### CockroachDB

CockroachDB 是 Spanner 的开源实现，目标是提供类似 Spanner 的分布式 SQL 数据库。

#### 与 Spanner 的区别

- 不依赖原子钟，使用 HLC（混合逻辑时钟）+ 事务重试来保证外部一致性
- 兼容 PostgreSQL 协议
- 完全开源，可部署在任意基础设施上
- 使用 Raft 共识替代 Paxos

---

## 4. 分布式缓存（Redis Cluster）

Redis Cluster 是 Redis 的分布式方案，提供数据分片和高可用。

### 架构

- **分片**：数据被划分为 16384 个哈希槽（Hash Slot），每个节点负责一部分槽
- **去中心化**：节点之间通过 Gossip 协议通信，无需中心化的代理
- **主从复制**：每个主节点可以有多个从节点，主节点故障时自动故障转移
- **客户端直连**：客户端根据槽映射表直接连接到正确的节点

### 关键机制

- **槽迁移**：支持在线迁移哈希槽，实现扩缩容
- **MOVED 重定向**：当客户端访问的槽不在当前节点时，返回 MOVED 指令
- **ASK 重定向**：迁移过程中，客户端可能需要临时访问目标节点
- **Gossip 协议**：节点间定期交换状态信息，检测故障

### 注意事项

- 不支持多键操作（除非键在同一个槽中，可通过 Hash Tag 实现）
- 不支持跨槽事务
- 故障转移期间可能丢失少量数据（异步复制）

---

## 5. MapReduce编程模型

MapReduce 由 Google 于 2004 年提出，是大数据批处理的奠基性编程模型。

### 执行流程

1. **Input Split**：将输入数据切分为若干分片（Split），每个分片分配给一个 Map 任务。
2. **Map 阶段**：每个 Map 任务读取一个分片，对每条记录执行 Map 函数，输出键值对 `<K, V>`。
3. **Shuffle 阶段**：框架对 Map 输出进行分区、排序、合并，相同 Key 的数据汇聚到同一个 Reduce 任务。
4. **Reduce 阶段**：每个 Reduce 任务对一组键值对执行 Reduce 函数，输出最终结果。
5. **Output**：结果写入分布式文件系统。

### 核心设计

- **数据本地化（Data Locality）**：Map 任务尽量在数据所在的节点执行，减少网络传输。
- **容错**：任务失败自动重试，Map 输出写入本地磁盘（中间结果可重算）。
- **简单编程接口**：用户只需实现 Map 和 Reduce 两个函数。

### 局限性

- 不适合迭代计算（每次迭代需要读写磁盘）
- 不适合实时处理（批处理模型，延迟高）
- 不适合复杂的 DAG 计算

---

## 6. 数据流系统

### Apache Spark

Spark 于 2014 年成为 Apache 顶级项目，是对 MapReduce 的重要改进。

#### 核心创新 — RDD

RDD（Resilient Distributed Dataset）是 Spark 的核心抽象：

- **不可变**：RDD 一旦创建不可修改，变换产生新的 RDD
- **分区**：数据分布在集群的不同节点上
- **血统（Lineage）**：记录 RDD 的变换历史，用于故障恢复（重新计算丢失的分区）
- **惰性求值**：变换操作只记录依赖关系，遇到行动操作才真正执行

#### 与 MapReduce 的对比

| 特性 | MapReduce | Spark |
|------|-----------|-------|
| 数据模型 | 键值对 | RDD / DataFrame / Dataset |
| 中间结果 | 写磁盘 | 内存（可选磁盘） |
| 迭代计算 | 每次读写 HDFS | 数据缓存在内存中 |
| DAG 支持 | 仅 Map → Reduce | 复杂 DAG |
| 延迟 | 秒到分钟 | 毫秒到秒 |

#### Spark 生态

- **Spark SQL**：结构化数据处理
- **Spark Streaming**：微批次流处理
- **Structured Streaming**：基于 DataFrame 的流处理
- **MLlib**：机器学习库
- **GraphX**：图计算

### Apache Flink

Flink 是真正的流处理框架，以"流处理优先"（Stream-First）为设计理念。

#### 核心特性

- **真正的流处理**：数据逐条处理，延迟更低（毫秒级）
- **事件时间处理**：支持基于事件时间的窗口计算和乱序数据处理
- **Exactly-Once 语义**：通过 Checkpoint 机制保证精确一次处理
- **状态管理**：内置大规模状态管理，支持多种状态后端（Memory、RocksDB）

#### Flink vs Spark Streaming

| 特性 | Flink | Spark Streaming |
|------|-------|-----------------|
| 处理模型 | 真正的流处理 | 微批次（Micro-Batch） |
| 延迟 | 毫秒级 | 秒级 |
| 事件时间 | 原生支持 | Structured Streaming 支持 |
| 状态管理 | 内置强大状态管理 | 较弱 |
| 窗口语义 | 丰富（滚动、滑动、会话、全局） | 有限 |

#### Checkpoint 与 Exactly-Once

- 基于 Chandy-Lamport 算法实现分布式快照
- Barrier 对齐机制保证数据只被处理一次
- 状态定期快照到持久化存储（HDFS、S3）
- 故障恢复时从最近的 Checkpoint 恢复状态

---

## 7. 分布式消息队列（Kafka架构）

Apache Kafka 是分布式消息系统的事实标准，由 LinkedIn 开发并于 2011 年开源。

### 架构组件

- **Broker**：Kafka 服务器节点，存储和转发消息
- **Topic**：消息的逻辑分类，一个 Topic 可以有多个 Partition
- **Partition**：消息的物理存储单元，有序、不可变、可追加写入
- **Producer**：消息生产者，将消息发送到指定 Topic
- **Consumer**：消息消费者，从 Topic 读取消息
- **Consumer Group**：消费者组，同一组内的消费者分摊 Partition 的消费
- **ZooKeeper / KRaft**：管理集群元数据和 Leader 选举（新版本使用 KRaft 去除 ZK 依赖）

### 关键设计

#### 高性能

- **顺序写磁盘**：Kafka 将消息追加写入 Partition，利用磁盘顺序写的高性能
- **零拷贝（Zero-Copy）**：使用 sendfile 系统调用直接从磁盘发送数据到网络，避免内核态与用户态之间的数据拷贝
- **批量处理**：Producer 批量发送消息，Consumer 批量拉取消息
- **页缓存（Page Cache）**：利用操作系统的页缓存，减少磁盘 IO

#### 持久化与可靠性

- 消息持久化到磁盘，通过副本机制保证高可用
- `acks` 参数控制写入确认策略：
  - `acks=0`：不等待确认，最高吞吐
  - `acks=1`：Leader 确认即可
  - `acks=all`：所有 ISR 副本确认，最高可靠
- ISR（In-Sync Replicas）：保持与 Leader 同步的副本集合

#### 消费模型

- Pull 模型：Consumer 主动从 Broker 拉取消息
- Offset 管理：Consumer 维护消费进度（Offset），支持回溯消费
- Rebalance：Consumer Group 内部的 Partition 分配策略

### Kafka 在分布式系统中的角色

- **解耦**：Producer 和 Consumer 不直接依赖
- **削峰**：缓冲突发的消息流量
- **事件驱动**：支撑事件驱动架构和 CQRS 模式
- **数据管道**：连接各种数据源和数据目标（Kafka Connect）
- **流计算**：与 Flink/Spark Streaming 集成进行实时计算

---

## 8. 微服务架构中的分布式问题

### 服务发现

微服务架构中，服务实例动态变化，需要服务发现机制：

| 方案 | 实现 | 特点 |
|------|------|------|
| 客户端发现 | Eureka、Nacos | 客户端查询注册中心，自行负载均衡 |
| 服务端发现 | Kubernetes Service | 平台提供负载均衡，客户端无需感知 |
| DNS 发现 | CoreDNS | 通过 DNS 记录发现服务，简单但灵活性低 |

### 配置管理

分布式配置需要解决动态更新和版本管理：

- **集中式配置**：Nacos、Apollo、Consul KV
- **推拉结合**：长轮询或 WebSocket 推送变更
- **版本与回滚**：配置变更需要记录历史，支持回滚
- **灰度发布**：配置可以按条件推送到部分实例

### 链路追踪

分布式追踪帮助理解请求在多个服务之间的调用路径：

- **OpenTracing / OpenTelemetry**：标准化的追踪 API
- **Jaeger / Zipkin**：分布式追踪系统
- **核心概念**：Trace（调用链）、Span（一个操作）、Context Propagation（上下文传播）
- **采样策略**：全量采样成本高，通常使用概率采样或限速采样

---

## 9. 服务网格（Istio/Envoy）

### 什么是服务网格

服务网格（Service Mesh）是处理服务间通信的专用基础设施层，将通信逻辑从业务代码中剥离。

### 数据平面 — Envoy

- **Sidecar 代理**：每个服务实例旁部署一个 Envoy 代理，所有进出流量经过代理
- **L4/L7 代理**：支持 TCP 和 HTTP 协议的流量管理
- **负载均衡**：支持多种负载均衡算法（轮询、最少连接、一致性哈希）
- **熔断器**：自动检测故障服务并切断流量
- **可观测性**：自动生成指标、日志和追踪数据

### 控制平面 — Istio

- **流量管理**：灰度发布、流量镜像、故障注入
- **安全**：mTLS 自动加密服务间通信、RBAC 访问控制
- **策略**：限流、访问控制策略
- **可观测性**：集成 Prometheus、Grafana、Jaeger

### Istio 架构

```
控制平面 (Istiod)
  ├── Pilot   — 服务发现和流量管理配置
  ├── Citadel — 证书管理和 mTLS
  └── Galley — 配置验证和分发

数据平面
  ├── Envoy Sidecar (Pod A) ←→ 业务容器 A
  └── Envoy Sidecar (Pod B) ←→ 业务容器 B
```

### 适用场景与权衡

| 优势 | 代价 |
|------|------|
| 业务代码完全不感知网络通信细节 | 引入 Sidecar 增加资源开销和延迟 |
| 统一的流量管理和安全策略 | 架构复杂度增加，运维成本上升 |
| 强大的可观测性能力 | 学习曲线陡峭 |
| 语言无关，支持多语言微服务 | 调试难度增加（多了一层代理） |

### 与 API 网关的关系

- **服务网格**：处理东西向（服务间）流量
- **API 网关**：处理南北向（外部到内部）流量
- 两者互补，Istio Ingress Gateway 可以充当 API 网关角色

---

> 回到目录：[60-分布式系统](../) — 查看完整的学习路线和主题索引。
