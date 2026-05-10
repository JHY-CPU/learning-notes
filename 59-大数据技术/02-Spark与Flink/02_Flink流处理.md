# Flink流处理


## Flink流处理


大数据Flink流处理


Apache Flink是以流处理为核心的分布式计算引擎，支持事件驱动的有状态计算。


## Flink架构


```
Flink 运行架构：
┌─────────────────────────────────────────────┐
│  JobManager (作业管理器)                      │
│  ├── 调度Task的执行                          │
│  ├── 管理检查点(Checkpoint)和恢复            │
│  ├── 协调分布式快照                          │
│  └── 处理Failover                           │
│                                             │
│  TaskManager (任务管理器)                     │
│  ├── 执行具体的Task                          │
│  ├── 管理Slot（资源槽位）                     │
│  └── 管理网络缓冲和数据交换                   │
│                                             │
│  Slot（任务槽）：                             │
│  ├── TaskManager的资源子集                   │
│  ├── 同一Job的不同Task可以共享Slot           │
│  └── 实现资源隔离和复用                       │
│                                             │
│  核心理念：                                  │
│  - 一切皆流：批处理是流处理的特殊情况          │
│  - 有状态计算：维护和更新中间状态              │
│  - 事件时间驱动：基于事件发生时间处理          │
└─────────────────────────────────────────────┘
```


## 流处理 vs 批处理


```
批处理 (Batch Processing)：
- 有界数据集（bounded data）
- 处理完整个数据集后输出结果
- 延迟：分钟~小时级
- 典型：MapReduce, Hive, Spark Batch

流处理 (Stream Processing)：
- 无界数据流（unbounded stream）
- 数据到来即处理，持续输出
- 延迟：毫秒~秒级
- 典型：Flink, Kafka Streams, Spark Streaming

三种处理语义：
1. At-most-once：可能丢失，不重发
2. At-least-once：不丢失，可能重复
3. Exactly-once：不丢失不重复（最理想）

Flink通过Checkpoint实现Exactly-once：
- 定期对所有算子状态做分布式快照
- 故障时从最近的Checkpoint恢复
- 配合两阶段提交实现端到端Exactly-once
```


## 事件时间与Watermark


```
时间语义：
┌─────────────────────────────────────────────┐
│  Event Time（事件时间）                       │
│  └── 数据产生的时间，嵌入在数据中              │
│  Processing Time（处理时间）                  │
│  └── 数据被处理时的系统时间                    │
│  Ingestion Time（摄入时间）                   │
│  └── 数据进入Flink的时间                      │
└─────────────────────────────────────────────┘

Watermark（水位线）机制：
- 解决事件时间下的乱序问题
- Watermark是一个时间戳，表示"此时间之前的数据都已到达"
- 当 Watermark >= Window的结束时间 时，触发窗口计算

示例：
事件：e1(t=1), e3(t=3), e2(t=2), e4(t=4)
Watermark策略：W(t) = max(EventTime) - 5秒

处理过程：
1. e1(t=1)到达 → W = 1-5 = -4
2. e3(t=3)到达 → W = 3-5 = -2
3. e2(t=2)到达 → W = 3-5 = -2（max不变）
4. e4(t=4)到达 → W = 4-5 = -1

允许乱序的时间 = Watermark延迟（5秒）
迟到数据处理：
- Allowed Lateness：窗口关闭后的容忍时间
- Side Output：将迟到数据输出到侧输出流
```


> **Note:** 合理设置Watermark延迟：太短会丢数据，太长会增加计算延迟。


## 状态管理


```
Flink 状态类型：
┌─────────────────────────────────────────────┐
│  Keyed State（键控状态）                      │
│  ├── ValueState<T>：单个值                  │
│  ├── ListState<T>：列表                     │
│  ├── MapState<K,V>：键值对                  │
│  ├── ReducingState<T>：聚合值               │
│  └── AggregatingState<I,O>：自定义聚合      │
│  → 只能在KeyedStream上使用                   │
│  → 每个Key对应独立的状态                     │
│                                             │
│  Operator State（算子状态）                   │
│  ├── ListState：算子级别的列表状态            │
│  ├── UnionListState：广播状态                │
│  └── BroadcastState：广播状态               │
│  → 每个算子实例对应一个状态                   │
└─────────────────────────────────────────────┘

状态后端 (State Backend)：
- HashMapStateBackend：状态存储在内存/RocksDB
- EmbeddedRocksDBStateBackend：状态存储在RocksDB
  → 支持增量Checkpoint，适合大状态场景

Checkpoint配置示例：
env.enableCheckpointing(60000);  // 60秒一次
env.getCheckpointConfig()
    .setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig()
    .setMinPauseBetweenCheckpoints(30000); // 最小间隔30秒
```


## Flink窗口操作


```
窗口类型：
┌─────────────────────────────────────────────┐
│  滚动窗口 (Tumbling Window)                  │
│  │---w1---│---w2---│---w3---│               │
│  固定大小，不重叠，如每5分钟一个窗口           │
│                                             │
│  滑动窗口 (Sliding Window)                   │
│  │------w1------|                           │
│      │------w2------|                       │
│          │------w3------|                   │
│  固定大小，有重叠，如每5分钟窗口，滑动步长1分钟│
│                                             │
│  会话窗口 (Session Window)                   │
│  |--s1--|  gap  |--s2--|                    │
│  按活动会话分组，gap超过阈值则分隔             │
│                                             │
│  全局窗口 (Global Window)                    │
│  所有数据进入同一窗口，需自定义Trigger         │
└─────────────────────────────────────────────┘

代码示例：
stream.keyBy(event -> event.userId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .allowedLateness(Time.minutes(1))        // 允许迟到1分钟
    .sideOutputLateData(lateOutputTag)       // 迟到数据侧输出
    .reduce((a, b) -> merge(a, b));
```


> **Note:** Flink是真正的流处理引擎（逐条处理），Spark Streaming是微批处理（攒一批再处理），延迟和语义都有差异。


<!-- Converted from: 02_Flink流处理.html -->
