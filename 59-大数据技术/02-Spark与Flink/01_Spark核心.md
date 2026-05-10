# Spark核心


## Spark核心


大数据SparkRDD


Apache Spark是新一代分布式计算框架，以内存计算为核心，比MapReduce快10-100倍。


## Spark架构概览


```
Spark 运行架构：
┌─────────────────────────────────────────────┐
│  Driver Program (驱动程序)                   │
│  ├── SparkContext：应用入口                  │
│  ├── DAG Scheduler：将Job划分为Stage         │
│  └── Task Scheduler：将Task分发到Executor    │
│                                             │
│  Cluster Manager (集群管理器)                 │
│  ├── Standalone（Spark自带）                 │
│  ├── YARN                                  │
│  ├── Mesos                                 │
│  └── Kubernetes                            │
│                                             │
│  Executor (执行器 - 每个Worker Node上运行)    │
│  ├── 执行具体的Task                         │
│  ├── 将数据缓存在内存中                      │
│  └── 向Driver汇报执行状态                   │
│                                             │
│  Task (任务)                                │
│  └── 最小的执行单元，对应一个Partition的计算  │
└─────────────────────────────────────────────┘
```


## RDD - 弹性分布式数据集


RDD（Resilient Distributed Dataset）是Spark的核心抽象，不可变的分布式对象集合。


```
RDD 五大属性：
1. 分区列表 (partitions)：数据被分成多个分区
2. 计算函数 (compute)：每个分区的计算逻辑
3. 依赖列表 (dependencies)：RDD之间的血缘关系
4. 分区器 (partitioner)：决定数据如何分区（Hash/Range）
5. 首选位置 (preferredLocations)：数据本地性优化

RDD 操作类型：
- Transformation（转换）：懒执行，构建DAG
  map, filter, flatMap, reduceByKey, join, union ...
- Action（行动）：触发实际计算
  count, collect, first, take, reduce, saveAsTextFile ...

容错机制：血缘(Lineage)
- RDD通过DAG记录所有转换操作
- 丢失数据时只需重算丢失的分区
- 不需要数据复制，通过重算恢复
```


> **Note:** RDD是只读的、不可变的。每次转换都产生新的RDD，原始RDD不变。


## 惰性求值与DAG执行


```
惰性求值 (Lazy Evaluation)：
- Transformation不会立即执行，只记录操作
- 只有遇到Action才触发真正计算
- 好处：Spark可以优化整个DAG，减少不必要的计算

DAG (Directed Acyclic Graph) 有向无环图：
- 每个RDD是一个节点
- 转换操作是节点之间的边
- DAG Scheduler将DAG切分为Stage

Stage划分：以Shuffle为边界
┌─────────────────────────────────────────┐
│  Job                                    │
│  ├── Stage 0 (ShuffleMapStage)          │
│  │   Task 0 ─┬─ Task 1 ─┬─ Task 2      │
│  │           └──────────┘               │
│  │             Shuffle                   │
│  ├── Stage 1 (ShuffleMapStage)          │
│  │   Task 0 ─┬─ Task 1                  │
│  │           └─────── Shuffle            │
│  └── Stage 2 (ResultStage)              │
│      Task 0 ─┬─ Task 1 ─┬─ Task 2      │
└─────────────────────────────────────────┘

宽依赖 vs 窄依赖：
- 窄依赖：父RDD每个分区最多被子RDD一个分区依赖（map, filter）
- 宽依赖：父RDD分区被子RDD多个分区依赖（reduceByKey, groupByKey）
- 宽依赖 = Shuffle操作 = Stage边界
```


## DataFrame与Spark SQL


```
DataFrame = RDD + Schema + Catalyst优化器
- 比RDD更高效（Tungsten列式存储+代码生成）
- 支持SQL查询
- 自动优化执行计划

Spark SQL 示例（PySpark）：

# 从文件创建DataFrame
df = spark.read.parquet("/data/users.parquet")

# DSL风格
df.filter(df.age > 18) \
  .groupBy("city") \
  .agg(avg("salary").alias("avg_salary")) \
  .orderBy(desc("avg_salary")) \
  .show()

# SQL风格
df.createOrReplaceTempView("users")
result = spark.sql("""
    SELECT city, AVG(salary) as avg_salary
    FROM users
    WHERE age > 18
    GROUP BY city
    ORDER BY avg_salary DESC
""")

Catalyst优化器：
1. 解析：SQL → 逻辑计划（Unresolved Logical Plan）
2. 分析：关联元数据 → 逻辑计划
3. 优化：谓词下推、列裁剪、常量折叠
4. 物理计划：选择最优执行策略
5. 代码生成：生成Java字节码执行
```


> **Note:** 优先使用DataFrame/Dataset而非RDD，性能更优且Catalyst优化器会自动优化执行计划。


## Spark生态系统


```
Spark 生态组件：
┌───────────────────────────────────────┐
│  Spark SQL   │  结构化数据处理          │
│  Spark Core  │  RDD基础引擎            │
│  Spark       │  实时微批处理            │
│  Streaming   │  (Structured Streaming) │
│  MLlib       │  机器学习库             │
│  GraphX      │  图计算                 │
└───────────────────────────────────────┘

Structured Streaming（结构化流）：
- 将流数据看作无限增长的表
- 使用与批处理相同的DataFrame API
- 基于Spark SQL引擎的增量执行
- 支持Exactly-once语义
- 输出模式：Complete / Append / Update
```


<!-- Converted from: 01_Spark核心.html -->
