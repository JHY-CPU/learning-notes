# YARN与资源调度


## YARN与资源调度


大数据YARN资源调度


YARN（Yet Another Resource Negotiator）是 Hadoop 2.x 的资源管理框架，实现了资源管理与计算框架的解耦。


## YARN 整体架构


YARN 采用经典的主从架构，将资源管理和作业调度/监控分离。


```
┌──────────────────────────────────────────────────────────────┐
│                      YARN 架构                                │
│                                                              │
│  ┌─────────────────────────────────────────────┐             │
│  │         ResourceManager (全局资源管理)        │             │
│  │  ┌─────────────────┐  ┌──────────────────┐  │             │
│  │  │ Scheduler       │  │ ApplicationsMgr  │  │             │
│  │  │ (资源调度器)     │  │ (应用管理器)      │  │             │
│  │  │ - 不负责监控     │  │ - 管理所有应用    │  │             │
│  │  │ - 纯粹的调度     │  │ - 启动/监控 AM    │  │             │
│  │  └─────────────────┘  └──────────────────┘  │             │
│  └──────────────────┬──────────────────────────┘             │
│                     │                                        │
│        ┌────────────┼────────────┐                           │
│        ▼            ▼            ▼                           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                      │
│  │  Node    │ │  Node    │ │  Node    │                      │
│  │  Manager │ │  Manager │ │  Manager │                      │
│  │ (从节点) │ │ (从节点) │ │ (从节点) │                      │
│  │ 管理本机 │ │ 管理本机 │ │ 管理本机 │                      │
│  │ 资源+容器│ │ 资源+容器│ │ 资源+容器│                      │
│  └──────────┘ └──────────┘ └──────────┘                      │
└──────────────────────────────────────────────────────────────┘
```


## ResourceManager 详解


ResourceManager 是 YARN 的全局资源管理者，负责整个集群的资源分配和调度。


```
ResourceManager 两大组件：

1. Scheduler（调度器）
   - 纯粹的资源调度，不负责应用的监控和状态跟踪
   - 根据容量、队列等限制条件分配资源
   - 基于 Container 的概念进行资源抽象
   - Container = {CPU, Memory, 磁盘, 网络} 等资源的封装

2. ApplicationMaster（应用管理器）
   - 接受作业提交
   - 为应用分配第一个 Container 来运行 ApplicationMaster
   - 负责在 ApplicationMaster 失败时重启它

ApplicationMaster (AM)：
- 每个应用拥有独立的 AM
- AM 向 ResourceManager 申请资源
- AM 与 NodeManager 通信来执行和监控任务
- AM 负责任务的容错和重试

HA（高可用）方案：
- Active/Standby 双 ResourceManager
- 通过 ZooKeeper 进行故障切换
- 使用 ZKStore 或 LevelDB 持久化状态
```


## NodeManager 详解


NodeManager 是每个工作节点上的资源和任务管理器。


```
NodeManager 职责：
┌─────────────────────────────────────────────────┐
│  NodeManager                                    │
│  ├── 管理单个节点上的资源                         │
│  ├── 启动和监控 Container                        │
│  ├── 定期向 RM 汇报心跳和资源使用情况              │
│  ├── 管理 Container 生命周期                     │
│  ├── 执行来自 AM 的命令（启动/停止任务）           │
│  └── 日志聚合（将 Container 日志上传到 HDFS）      │
│                                                 │
│  Container 概念：                                │
│  ├── 逻辑资源的抽象封装                           │
│  ├── 包含 CPU 核数、内存大小等                     │
│  ├── 是任务执行的最小单位                         │
│  └── 由 RM 分配，NM 管理执行                     │
│                                                 │
│  资源模型：                                      │
│  ├── 可配置的：memory-mb, vcores                  │
│  ├── 最小/最大容器内存                             │
│  ├── 资源计算器（DefaultResourceCalculator）       │
│  └── DominantResourceCalculator（DRF 多维资源）   │
└─────────────────────────────────────────────────┘
```


## YARN 调度策略


YARN 支持多种调度策略，适用于不同的集群管理需求。


```
1. FIFO Scheduler（先进先出调度器）
   ┌─────────────────────────────────────┐
   │  Queue: [Job1] [Job2] [Job3]       │
   │  按提交顺序依次执行                    │
   │  优点：简单，无额外开销                │
   │  缺点：大作业阻塞、不适合共享集群       │
   └─────────────────────────────────────┘

2. Capacity Scheduler（容量调度器）
   ┌─────────────────────────────────────┐
   │  队列A (60%) │ 队列B (30%) │ 队列C (10%) │
   │  [Job1][Job2]│ [Job3]      │ [Job4]      │
   │  每个队列保证最小容量                    │
   │  空闲容量可被其他队列借用                │
   │  支持多租户，Yahoo 开发                 │
   └─────────────────────────────────────┘

3. Fair Scheduler（公平调度器）
   ┌─────────────────────────────────────┐
   │  所有作业公平共享集群资源               │
   │  Job1: 50% │ Job2: 50%              │
   │  短作业可快速获取资源                   │
   │  支持权重、最小份额保证                │
   │  Facebook 开发                       │
   └─────────────────────────────────────┘

对比总结：
┌────────────┬──────────┬──────────────┬──────────┐
│            │ FIFO     │ Capacity     │ Fair     │
├────────────┼──────────┼──────────────┼──────────┤
│ 设计目标   │ 简单高效   │ 多租户容量保证 │ 公平共享  │
│ 适用场景   │ 专用集群   │ 多部门共享    │ 多用户混合 │
│ 队列隔离   │ 无        │ 强隔离        │ 动态平衡  │
│ 饥饿问题   │ 大作业阻塞 │ 较少          │ 基本消除  │
└────────────┴──────────┴──────────────┴──────────┘
```


> **Note:** 生产环境中 Capacity 和 Fair Scheduler 使用最为广泛，Capacity 适合多租户严格隔离场景，Fair 适合弹性共享场景。


## 资源分配与抢占机制


```
资源分配流程：
1. 客户端提交应用 → RM 的 ApplicationsManager
2. RM 分配第一个 Container → 启动 ApplicationMaster
3. AM 向 RM 注册并申请资源（ResourceRequest）
4. RM 分配 Container → AM 通知对应 NM 启动任务
5. NM 启动 Container → 执行任务 → 汇报状态
6. AM 监控任务进度 → 完成后释放资源

资源抢占（Preemption）：
- 当某队列长期得不到应有资源份额时触发
- 抢占策略：
  * Kill 直接终止任务
  * Graceful 通知 AM 优雅释放
- 配置：yarn.scheduler.fair.preemption=true
- 抢占窗口：等待时间后执行抢占

资源估算：
- 最小容器：yarn.scheduler.minimum-allocation-mb = 1024MB
- 最大容器：yarn.scheduler.maximum-allocation-mb = 8192MB
- 节点总资源：yarn.nodemanager.resource.memory-mb
```


<!-- Converted from: 03_YARN与资源调度.html -->
