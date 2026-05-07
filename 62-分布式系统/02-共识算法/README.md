# 02-共识算法

> 共识算法是分布式系统的核心，它使得一组节点能够在存在故障的情况下对某个值达成一致。本章系统梳理经典共识算法的原理与工程实践。

---

## 1. 共识问题的定义

分布式共识是指多个节点就某个值（或一系列值）达成一致的过程。形式化定义中，共识算法需要满足以下性质：

### 四大基本性质

1. **一致性（Agreement / Safety）**：所有正确的节点最终决定相同的值。不存在两个正确的节点做出不同的决定。
2. **有效性（Validity / Non-triviality）**：如果所有正确的节点提议同一个值 v，那么最终决定的值必须是 v。算法不能凭空产生一个值。
3. **可终止性（Termination / Liveness）**：所有正确的节点最终都会做出决定。不会永远等待。
4. **容错性（Fault Tolerance）**：系统在部分节点故障的情况下仍然能够达成共识。

### FLP的阴影

如前一章所述，FLP 不可能性定理表明在纯异步系统中不存在确定性的共识算法。因此实际算法通过超时、领导者机制等方式引入"部分同步"假设来规避这一限制。

---

## 2. Paxos算法

Paxos 由 Leslie Lamport 于 1989 年提出（1998 年正式发表），是最经典的共识算法，也是分布式系统领域最重要的理论贡献之一。

### 角色定义

- **Proposer（提议者）**：提出提案（Proposal），包含提案编号和值。
- **Acceptor（接受者）**：对提案进行投票，接受或拒绝提案。
- **Learner（学习者）**：学习已经被选定的值，不做决策。

### Basic Paxos

#### Phase 1 — Prepare 阶段

1. Proposer 选择一个全局唯一的递增提案编号 n，向多数派 Acceptor 发送 `Prepare(n)` 请求。
2. Acceptor 收到 `Prepare(n)` 后：
   - 如果 n 大于它之前响应过的所有 Prepare 请求的编号，则承诺不再接受编号小于 n 的提案，并回复它之前已经接受过的编号最大的提案（如果有）。
   - 否则忽略该请求。

#### Phase 2 — Accept 阶段

1. Proposer 收到多数派 Acceptor 的 Prepare 响应后：
   - 如果所有响应中都没有已接受的提案，则 Proposer 可以自由决定提案的值。
   - 如果有已接受的提案，则选择其中编号最大的那个值。
   - 向多数派 Acceptor 发送 `Accept(n, v)` 请求。
2. Acceptor 收到 `Accept(n, v)` 后：
   - 如果 n 不小于它承诺的最小提案编号，则接受该提案并回复。
   - 否则拒绝。

#### Learn 阶段

一旦多数派 Acceptor 接受了同一个提案，该值即被选定（Chosen）。Learner 通过监听 Acceptor 的响应学习被选定的值。

### Multi-Paxos

Basic Paxos 每次达成共识都需要两轮通信，效率较低。Multi-Paxos 通过选举一个**稳定的领导者**来优化：

- 第一次选举领导者时仍需完整的两阶段流程
- 选举成功后，后续提案可以跳过 Prepare 阶段，直接执行 Accept 阶段
- 将两轮通信降为一轮，显著提高了吞吐量
- 需要处理领导者变更时的状态恢复

### Paxos的活锁问题

Basic Paxos 中存在活锁（Livelock）的可能：当多个 Proposer 交替提高提案编号并发送 Prepare 请求时，任何提案都无法获得多数派接受。

**解决方案**：引入领导者选举机制，确保同一时刻只有一个活跃的 Proposer。Multi-Paxos 实质上解决了这个问题。

---

## 3. Raft算法

Raft 由 Diego Ongaro 和 John Ousterhout 于 2014 年提出，设计目标是比 Paxos 更易于理解和实现。Raft 已被 etcd、Consul、CockroachDB 等广泛采用。

### 核心思想：分解问题

Raft 将共识问题分解为三个相对独立的子问题：

1. **领导者选举（Leader Election）**
2. **日志复制（Log Replication）**
3. **安全性保证（Safety）**

### 节点角色

| 角色 | 说明 |
|------|------|
| Leader | 处理所有客户端请求，负责日志复制 |
| Follower | 被动接收 Leader 的日志复制请求 |
| Candidate | 选举过程中的临时角色，用于发起投票 |

### Leader选举

#### Term（任期）

Raft 将时间划分为连续的 Term，每个 Term 以一次选举开始：

- 每个 Term 最多有一个 Leader
- Term 是逻辑时钟，用于检测过期的信息
- 节点发现自己的 Term 小于对方时，立即更新为对方的 Term 并转变为 Follower

#### 选举超时机制

1. Follower 在选举超时时间内未收到 Leader 的心跳，则转变为 Candidate，Term + 1，向所有节点发送 `RequestVote` 请求。
2. 每个节点在每个 Term 内只能投一票（先到先得）。
3. Candidate 获得多数派投票后成为 Leader，立即开始发送心跳。
4. 如果收到更高 Term 的 Leader 心跳，Candidate 回退为 Follower。

**随机化超时**：为了避免多个 Candidate 同时发起选举导致选票分裂，每个节点的选举超时时间在 150ms-300ms 之间随机化。

### 日志复制

1. 客户端请求到达 Leader，Leader 将其作为新条目追加到本地日志。
2. Leader 通过 `AppendEntries` RPC 将新条目复制到所有 Follower。
3. 当条目被多数派复制后，Leader 将该条目**提交（Commit）**，并应用到状态机。
4. Leader 在后续的 `AppendEntries` 中通知 Follower 已提交的条目索引，Follower 随后也提交并应用。

#### 日志匹配属性（Log Matching Property）

Raft 保证以下两个性质：

- 如果两个日志条目具有相同的索引和 Term，则它们存储相同的命令。
- 如果两个日志条目具有相同的索引和 Term，则它们之前的所有条目也都相同。

通过 `AppendEntries` 中的一致性检查（包含前一条日志的索引和 Term）来保证。

### 安全性保证

#### 选举限制（Election Restriction）

Candidate 的 `RequestVote` 请求中包含其最后一条日志的索引和 Term。投票节点只有在 Candidate 的日志至少和自己一样新时才投出选票。这保证了被选出的 Leader 一定拥有所有已提交的日志条目。

#### 提交规则

Leader 只能提交当前 Term 的日志条目为多数派复制的状态。不能通过复制旧 Term 的条目并使其被多数派副本来直接提交，必须通过提交当前 Term 的条目间接提交。

### 成员变更

Raft 支持集群成员的动态变更，保证变更过程中不会出现两个 Leader。

#### Joint Consensus（联合共识）

- 两阶段方法：先切换到中间配置（旧+新），再切换到新配置
- 在中间配置中，决策需要在旧配置和新配置中都获得多数派
- 实现复杂，较少被实际采用

#### 单服务器变更（Single Server Change）

- 每次只添加或移除一个节点
- 新旧配置的多数派必然存在交集，不会产生两个 Leader
- 实现简单，是大多数 Raft 实现的默认方式

### 日志压缩（Snapshot）

随着时间推移，日志会无限增长。Raft 使用快照（Snapshot）进行日志压缩：

1. 每个节点独立地将已提交的状态机状态写入快照。
2. 快照包含元数据：最后包含的索引和 Term、最新配置。
3. 如果 Leader 需要向落后太多的 Follower 发送日志，改为发送快照（`InstallSnapshot` RPC）。
4. 快照后的日志条目可以安全丢弃。

---

## 4. ZAB协议（ZooKeeper Atomic Broadcast）

ZAB 是 Apache ZooKeeper 使用的原子广播协议，与 Raft 类似但设计背景不同。ZAB 专为 ZooKeeper 的主从复制模型设计。

### 三个阶段

#### 发现阶段（Discovery / Leader Election）

- 所有节点参与选举，选出一个 Leader
- Leader 收集所有 Follower 上已提交的事务，确定最新的历史
- 使用 Fast Leader Election 算法（类似 Raft 的投票机制）

#### 同步阶段（Synchronization）

- Leader 将自己的事务历史发送给所有 Follower
- Follower 根据 Leader 的事务更新自己的状态
- 确保 Leader 和所有 Follower 在广播开始前状态一致
- 同步完成后，集群进入广播阶段

#### 广播阶段（Broadcast）

- Leader 接收客户端的写请求，生成事务提案
- 使用类似 2PC 的两阶段协议进行广播：
  1. Leader 发送 `PROPOSE` 消息给所有 Follower
  2. Follower 将提案追加到日志并回复 `ACK`
  3. Leader 收到多数派 `ACK` 后，发送 `COMMIT` 消息
  4. 所有节点执行该事务

### ZAB 与 Raft 的关键区别

| 特性 | ZAB | Raft |
|------|-----|------|
| 日志编号 | 使用 ZXID（64位：epoch + counter） | 使用 Term + Index |
| 恢复协议 | 有独立的发现和同步阶段 | 依赖选举限制保证 |
| 日志格式 | 定长 ZXID，高效比较 | 两个字段比较 |
| 客户端模型 | 支持读请求在 Follower 上处理 | 默认所有请求走 Leader |

---

## 5. 视图复制（Viewstamped Replication）

Viewstamped Replication（VR）由 Barbara Liskov 于 1988 年提出，是最先解决状态机复制问题的协议之一。

### 核心概念

- **View（视图）**：对应 Raft 的 Term，每个视图有一个主节点（Primary）
- **视图切换**：当主节点失效时，备份节点协作选出新主节点并进入新视图
- **正常操作**：主节点接收请求，排序后复制到备份节点

### 与 Raft 的关系

Raft 可以看作是 VR 的重新表述和改进，两者在本质上等价。Raft 的贡献在于更清晰地定义了 Leader 选举、日志复制和安全性之间的关系。

---

## 6. 共识算法的对比

| 算法 | 提出时间 | 复杂度 | 容错模型 | 通信轮次 | 工程采用 |
|------|----------|--------|----------|----------|----------|
| Paxos | 1989 | 高 | 崩溃故障（CFT） | 2轮（Basic）/ 1轮（Multi） | Chubby、Megastore |
| Raft | 2014 | 中 | 崩溃故障（CFT） | 1-2轮 | etcd、Consul、TiKV |
| ZAB | 2008 | 中 | 崩溃故障（CFT） | 2轮 | ZooKeeper |
| VR | 1988 | 中 | 崩溃故障（CFT） | 1轮 | 概念影响众多系统 |

### 共同点

- 都依赖多数派（Quorum）机制
- 都需要一个稳定的领导者提高效率
- 都通过日志复制实现状态机复制
- 都能容忍 f 个节点故障（总节点数 n ≥ 2f + 1）

---

## 7. 工程实践中的选型

### etcd — 选择 Raft

- **背景**：Kubernetes 的核心存储组件，存储集群配置和状态
- **为什么选 Raft**：
  - 论文清晰、伪代码完整，社区有大量成熟的 Go 实现（etcd/raft 库）
  - 线性一致性读的实现相对简单（ReadIndex 或 LeaseRead）
  - 成员变更使用单服务器变更，简单可靠
- **工程优化**：
  - ReadIndex 优化读性能
  - LeaseRead 进一步减少读延迟
  - PreVote 防止网络分区恢复后的无效选举

### ZooKeeper — 选择 ZAB

- **背景**：Apache 顶级项目，广泛用于服务发现、配置管理、分布式锁
- **为什么选 ZAB**：
  - ZAB 专门为 ZooKeeper 的数据模型（ZNode 树）设计
  - ZXID 比较效率高（单一 64 位整数 vs Raft 的 Term + Index 两字段）
  - Follower 可以处理读请求（通过 Sync 机制保证一致性）
- **与 Raft 的关系**：
  - ZAB 和 Raft 在理论上等价，都是 Paxos 的简化和工程化
  - ZooKeeper 团队在 2011 年的论文中详细比较了 ZAB 与 Paxos/Raft

### 选型建议

| 场景 | 推荐方案 |
|------|----------|
| 新项目，需要成熟共识库 | Raft（etcd/raft、hashicorp/raft） |
| 需要强一致键值存储 | etcd（K8s 生态）或 ZooKeeper |
| 需要线性一致性读 | Raft（ReadIndex）或 etcd |
| 需要 Watch/通知机制 | ZooKeeper（Watches 原生支持） |
| 需要高吞吐日志复制 | Multi-Raft / Parallel Raft |

---

> 下一章：[03-分布式事务](../03-分布式事务/) — 探讨跨节点事务的协调与一致性保证。
