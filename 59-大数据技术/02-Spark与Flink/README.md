# 02-Spark与Flink

## 1. Spark概述与MapReduce的对比

Apache Spark 是一个统一的分布式计算引擎，最初由UC Berkeley的AMPLab开发，旨在克服MapReduce的性能瓶颈。

**Spark vs MapReduce 对比：**

| 特性 | MapReduce | Spark |
|------|-----------|-------|
| 执行模型 | 每个Job输出写HDFS | 基于内存计算，DAG执行 |
| 延迟 | 高（秒级到分钟级） | 低（毫秒级） |
| 迭代计算 | 每次迭代写磁盘，效率极低 | 数据缓存在内存中，速度提升10-100倍 |
| 编程模型 | 仅Map/Reduce | 丰富的算子（map、filter、join、reduceByKey等） |
| 数据流 | Map → Disk → Reduce | DAG流水线执行 |
| 语言支持 | Java为主 | Scala、Java、Python、R、SQL |
| 生态 | 仅批处理 | 批处理、流处理、SQL、ML、Graph统一 |

Spark 的核心优势：**基于内存的迭代计算** 和 **统一的编程范式**。

---

## 2. RDD（弹性分布式数据集）

RDD（Resilient Distributed Dataset）是Spark最基础的数据抽象。

### 2.1 创建RDD、转换操作、行动操作

**创建RDD的方式：**
- 从集合创建：`sc.parallelize([1, 2, 3, 4])`
- 从文件创建：`sc.textFile("hdfs://path/to/file")`
- 从其他RDD转换而来

**转换操作（Transformation）——惰性求值：**
- `map(func)`：对每个元素应用函数
- `filter(func)`：过滤满足条件的元素
- `flatMap(func)`：一对多的映射
- `mapPartitions(func)`：对每个分区应用函数
- `sample(withReplacement, fraction, seed)`：采样
- `union(otherDataset)`：合并两个RDD
- `distinct()`：去重
- `groupByKey()`：按key分组
- `reduceByKey(func)`：按key聚合
- `sortByKey()`：按key排序
- `join(otherDataset)`：连接两个RDD

**行动操作（Action）——触发实际计算：**
- `collect()`：收集所有元素到Driver端
- `count()`：计算元素个数
- `first()` / `take(n)`：取前n个元素
- `reduce(func)`：聚合
- `foreach(func)`：对每个元素执行函数
- `saveAsTextFile(path)`：保存到文件
- `countByKey()`：统计每个key的个数

### 2.2 惰性求值与DAG

Spark采用 **惰性求值（Lazy Evaluation）** 机制：

- 转换操作（Transformation）不会立即执行，只是记录操作的血统关系
- 只有遇到行动操作（Action）时才触发实际计算
- Spark会构建 **DAG（有向无环图）** 表示计算依赖关系
- DAG Scheduler 将DAG划分为多个 **Stage**
- Stage内部是Pipeline流水线执行，Stage之间需要Shuffle

**Stage划分依据**：宽依赖（Shuffle Dependency）是Stage的边界
- **窄依赖（Narrow Dependency）**：子分区只依赖父RDD的一个分区，可Pipeline执行
- **宽依赖（Wide Dependency）**：子分区依赖父RDD的所有分区，需要Shuffle

### 2.3 容错与血统（Lineage）

RDD通过 **血统（Lineage）** 实现容错：

- 每个RDD记录其从数据源到当前状态的完整变换链
- 当某个分区数据丢失时，Spark可根据血统关系重新计算该分区
- 无需数据复制（Checkpoint除外），节省存储空间
- 血统本质上是一个DAG图，记录RDD之间的依赖关系

**Checkpoint机制**：
- 对于长血统链，Checkpoint可将中间结果持久化到可靠存储（如HDFS）
- 切断血统链，避免过长的重计算
- 适用于迭代计算和宽依赖较多的场景

---

## 3. Spark SQL与DataFrame

Spark SQL是Spark处理结构化数据的模块：

- **DataFrame**：以命名列组织的分布式数据集，类似关系型数据库表或Pandas DataFrame
- **Dataset**（Scala/Java）：类型安全的DataFrame，提供编译时类型检查
- 支持从多种数据源读取：JSON、Parquet、ORC、JDBC、Hive表等
- 支持标准SQL查询和DSL（DataFrame API）

**Catalyst优化器**：
- SQL/DataFrame经过Catalyst优化器生成优化的执行计划
- 包含：分析（Analysis）、逻辑优化（Logical Optimization）、物理规划（Physical Planning）、代码生成（Code Generation）
- 自动进行谓词下推、列裁剪、常量折叠等优化

**Tungsten执行引擎**：
- 基于内存数据结构和代码生成优化执行性能
- 使用紧凑的二进制行格式（UnsafeRow），减少GC开销
- 全阶段代码生成（Whole-Stage Code Generation）消除虚函数调用

---

## 4. Spark Streaming（微批处理）

Spark Streaming 是 Spark 的流处理组件，采用 **微批处理（Micro-Batch）** 模型：

- 将连续的数据流按时间间隔（如1秒）切分为小批次（DStream）
- 每个批次作为一个RDD进行处理
- DStream（Discretized Stream）本质上是一系列连续的RDD

**核心抽象**：
- **DStream**：离散化的数据流，底层是一系列RDD
- **窗口操作（Window Operations）**：跨越多个批次的聚合，需要指定窗口长度和滑动间隔
- **有状态转换**：`updateStateByKey` 维护跨批次的状态

**Structured Streaming**（Spark 2.x+）：
- 基于DataFrame/Dataset的流处理API
- 将流数据视为一个无限增长的表
- 支持事件时间处理和水印（Watermark）机制
- 语义上保证 Exactly-Once

**局限性**：
- 微批处理延迟通常在秒级，不适合低延迟场景
- 批次间隔越小，调度开销越大

---

## 5. Spark MLlib

MLlib 是 Spark 的机器学习库：

- **主要模块**：
  - 特征工程：TF-IDF、Word2Vec、标准化、归一化、特征选择
  - 分类：逻辑回归、决策树、随机森林、朴素贝叶斯、SVM
  - 回归：线性回归、决策树回归、随机森林回归
  - 聚类：K-Means、LDA、高斯混合模型
  - 协同过滤：ALS（交替最小二乘法）
  - 频繁项集：FP-Growth

- 基于DataFrame API（ML Pipeline），统一了数据处理和模型训练流程
- 支持模型的持久化和加载

---

## 6. Spark架构（Driver、Executor、Cluster Manager）

Spark 采用经典的主从架构：

- **Driver Program**：
  - 包含应用程序的 `main()` 函数和SparkContext
  - 创建RDD、执行转换操作、构建DAG
  - 将DAG划分为Stage，将Task调度到Executor执行
  - 管理整个应用的生命周期

- **Executor**：
  - 运行在工作节点上的进程，负责执行Task
  - 每个Executor有多个CPU核心（slot）和固定内存
  - 将计算结果返回给Driver或存储在内存/磁盘中
  - 应用之间Executor不共享

- **Cluster Manager**：
  - 负责集群资源的分配和管理
  - 支持：Standalone（自带）、YARN、Mesos、Kubernetes

- **部署模式**：
  - Client模式：Driver运行在提交任务的客户端
  - Cluster模式：Driver运行在集群中（更常用于生产环境）

---

## 7. 内存计算与缓存策略

Spark的核心优势是内存计算，缓存策略至关重要：

- **persist(StorageLevel)** / **cache()**：
  - `cache()` = `persist(StorageLevel.MEMORY_ONLY)`
  - 多种存储级别：
    - `MEMORY_ONLY`：仅内存，内存不足时不缓存
    - `MEMORY_AND_DISK`：内存优先，溢出到磁盘
    - `MEMORY_ONLY_SER`：序列化后存储在内存（更省内存但CPU开销大）
    - `DISK_ONLY`：仅磁盘
    - `OFF_HEAP`：堆外内存

- **何时缓存**：
  - RDD被多次使用时
  - 迭代计算中需要复用的中间结果
  - Shuffle操作后的数据

- **淘汰策略**：LRU（最近最少使用）

---

## 8. Flink概述

Apache Flink 是一个分布式流处理框架，以 **有状态流处理** 为核心设计理念：

- 真正的流处理引擎（不是微批处理），延迟可达毫秒级
- 批处理是流处理的特例（有界数据流）
- 高吞吐、低延迟、精确一次（Exactly-Once）语义

**架构组件**：
- **JobManager**：协调分布式执行，调度Task、协调Checkpoint和故障恢复
- **TaskManager**：执行Task的工作进程，每个TaskManager有多个Task Slot
- **Dispatcher**：提供REST接口，提交应用

---

## 9. DataStream API

Flink 的 DataStream API 是核心编程接口：

```java
// Source → Transformation → Sink 模式
StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

DataStream<String> source = env.addSource(new FlinkKafkaConsumer<>(...));
DataStream<Event> events = source
    .map(value -> parseEvent(value))
    .filter(event -> event.isValid())
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new MyAggregator());

events.addSink(new ElasticsearchSink<>(...));
env.execute("My Flink Job");
```

关键算子：
- `map`、`flatMap`、`filter`：基本转换
- `keyBy`：按key分区
- `window`：窗口操作
- `connect`、`union`：多流合并
- `process`：底层处理函数，可访问状态和定时器

---

## 10. 有状态流处理

Flink的核心竞争力在于有状态流处理：

- **Keyed State**：与某个key绑定的状态，由Flink自动管理
  - `ValueState<T>`：单个值
  - `ListState<T>`：列表
  - `MapState<K, V>`：键值映射
  - `ReducingState<T>` / `AggregatingState`：聚合状态

- **Operator State**：与算子实例绑定，不按key分组

- **状态后端（State Backend）**：
  - `HashMapStateBackend`：状态存储在TaskManager JVM堆内存
  - `EmbeddedRocksDBStateBackend`：状态存储在RocksDB（支持增量Checkpoint）

- **Checkpoint**：定期将状态快照持久化到可靠存储，支持故障恢复
- **Savepoint**：手动触发的全局一致性快照，用于版本升级和应用迁移

---

## 11. 时间语义

Flink 支持三种时间语义：

| 时间语义 | 说明 | 适用场景 |
|----------|------|----------|
| **Event Time** | 事件实际产生的时间 | 最常用，结果确定性最好 |
| **Processing Time** | 事件到达Flink处理引擎的系统时间 | 延迟最低，但结果不确定性 |
| **Ingestion Time** | 事件进入Flink Source的时间 | 折中方案 |

**Watermark（水印）**：
- 用于Event Time场景中处理乱序事件
- Watermark是一个时间戳，表示"此时间之前的所有事件都已到达"
- Watermark推进触发窗口计算
- `allowedLateness` 允许处理迟到的数据
- **侧输出（Side Output）** 可收集窗口关闭后仍然到达的迟到数据

---

## 12. 窗口机制

Flink提供了丰富的窗口类型：

- **滚动窗口（Tumbling Window）**：
  - 固定大小、不重叠
  - 例：每5分钟一个窗口

- **滑动窗口（Sliding Window）**：
  - 固定大小、可重叠
  - 需指定窗口大小和滑动步长
  - 例：每1分钟统计过去5分钟的数据

- **会话窗口（Session Window）**：
  - 根据活动间隙（Gap）划分
  - 连续活动属于同一会话，超过Gap时间则开新会话
  - 适用于用户行为分析

- **全局窗口（Global Window）**：
  - 所有元素归入同一个窗口
  - 需要自定义Trigger触发计算

**窗口函数**：
- `ReduceFunction` / `AggregateFunction`：增量聚合
- `ProcessWindowFunction`：可访问窗口元数据的全量处理

---

## 13. Exactly-Once语义

Flink通过 **分布式快照（Distributed Snapshots）** 实现Exactly-Once语义：

- **Chandy-Lamport算法**的变体
- JobManager周期性向所有Source注入 **Checkpoint Barrier**
- Barrier随数据流向下游传播
- 算子接收到所有输入流的Barrier后，执行本地状态快照
- 快照异步上传到持久化存储（如HDFS、S3）
- 故障恢复时，从最近的Checkpoint恢复状态

**端到端Exactly-Once**需要：
- Source支持重放（如Kafka可重置offset）
- Sink支持事务写入（如Two-Phase Commit、幂等写入）

---

## 14. Spark vs Flink对比

| 维度 | Spark | Flink |
|------|-------|-------|
| 核心模型 | 批处理优先，流处理是微批 | 流处理优先，批处理是特例 |
| 延迟 | 秒级（微批） | 毫秒级 |
| 时间语义 | Event Time（Structured Streaming） | Event Time、Processing Time、Ingestion Time |
| 状态管理 | 有限（Structured Streaming支持） | 一等公民，丰富的State API |
| 窗口支持 | 基于处理时间为主 | 丰富的窗口类型 |
| Exactly-Once | 支持（Structured Streaming） | 原生支持，端到端方案成熟 |
| 生态成熟度 | 生态丰富，社区庞大，工具链完善 | 流处理领域快速成长 |
| 批处理性能 | 内存计算优化成熟 | 批处理持续优化中 |
| SQL支持 | Spark SQL成熟 | Flink SQL快速发展 |
| 适用场景 | 批处理、ETL、交互分析、ML | 实时流处理、CEP、事件驱动应用 |

**选型建议**：
- 以批处理和数据分析为主 → Spark
- 以实时流处理和低延迟为主 → Flink
- 两者可结合使用：Spark做离线处理，Flink做实时处理（Lambda/Kappa架构）
