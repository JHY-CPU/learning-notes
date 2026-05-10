# Paxos与Raft


## Paxos与Raft共识算法


分布式共识Raft


共识算法解决分布式系统中多个节点对某个值达成一致的问题。


## Paxos算法


```
Paxos（Lamport 1990年，1998年发表）：
在异步网络中达成共识的经典算法

角色：
- Proposer（提议者）：提出提案
- Acceptor（接受者）：投票接受/拒绝提案
- Learner（学习者）：学习被接受的值

Basic Paxos 两个阶段：
┌─────────────────────────────────────────────┐
│  Phase 1: Prepare（准备阶段）                 │
│                                             │
│  Proposer → Acceptors:                      │
│    Prepare(n)  // n为提案编号，递增          │
│                                             │
│  Acceptor → Proposer:                       │
│    Promise(n, accepted_n, accepted_v)       │
│    // 承诺不再接受编号 < n 的提案             │
│    // 返回已接受的最高编号提案                │
│                                             │
│  Phase 2: Accept（接受阶段）                  │
│                                             │
│  Proposer → Acceptors:                      │
│    Accept(n, v)  // v为提议值               │
│    // 如果Phase1返回了已接受的值，使用最高    │
│    // 编号的那个值；否则使用自己的值          │
│                                             │
│  Acceptor → Proposer/Learner:               │
│    Accepted(n, v)                           │
│    // 多数派接受后，值被选定                  │
└─────────────────────────────────────────────┘

关键保证：
- 多数派交集：任意两个多数派至少有一个共同Acceptor
- 提案编号递增：保证新提案优先级更高
- 一个值一旦被选定，后续提案必须保持该值

Paxos变体：
- Multi-Paxos：选出Leader，省去Prepare阶段
- Fast-Paxos：在无冲突时跳过Phase1
```


## Raft算法


Raft是为可理解性而设计的共识算法（2014年），将共识分解为三个子问题。


```
Raft 三个子问题：
1. Leader选举 (Leader Election)
2. 日志复制 (Log Replication)
3. 安全性 (Safety)

角色：
- Leader：处理所有客户端请求，复制日志
- Follower：被动接收Leader的日志
- Candidate：选举过程中的临时角色

Leader选举流程：
1. Follower超时未收到心跳 → 变为Candidate
2. Term + 1，投票给自己，向其他节点请求投票
3. 收到多数派投票 → 成为Leader
4. 收到更高Term的Leader心跳 → 回到Follower
5. 选举超时（随机150-300ms）→ 重新选举

日志复制流程：
1. 客户端发送命令到Leader
2. Leader将命令追加到本地日志
3. Leader并行向所有Follower发送AppendEntries RPC
4. 多数派写入后，Leader提交(commit)日志
5. Leader通知Follower提交
6. Follower应用日志到状态机

安全性保证：
- 选举限制：Candidate的日志必须 >= 多数派
- 提交规则：Leader只能提交当前Term的日志条目
- 日志匹配：相同Index和Term的日志条目相同
```


## ZAB协议 (ZooKeeper Atomic Broadcast)


```
ZAB是ZooKeeper使用的共识协议，与Raft非常相似

两种模式：
1. 恢复模式（选举Leader）
2. 广播模式（复制日志）

与Raft的区别：
┌──────────┬──────────────┬──────────────────┐
│ 对比      │ Raft          │ ZAB               │
├──────────┼──────────────┼──────────────────┤
│ 选举      │ 随机超时      │ FastLeaderElection│
│ 日志      │ 连续索引      │ zxid（64位）       │
│ 术语      │ Term          │ epoch             │
│ 提交      │ 多数派提交    │ 多数派ack后提交     │
│ 快照      │ Snapshot      │ 快照+事务日志      │
└──────────┴──────────────┴──────────────────┘

ZAB的zxid设计：
- 高32位：epoch（选举轮次）
- 低32位：counter（每轮内的递增计数）
- 比较规则：先比较epoch，再比较counter

实际应用：
- etcd → Raft
- ZooKeeper → ZAB
- Consul → Raft
- TiKV → Raft
```


> **Note:** Raft因其可理解性成为工业界最广泛使用的共识算法，etcd、Consul、TiKV等都基于Raft。


<!-- Converted from: 01_Paxos与Raft.html -->
