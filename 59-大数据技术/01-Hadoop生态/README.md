# 01-Hadoop生态

## 1. 大数据概述

大数据是指无法用传统数据库工具在合理时间内完成采集、存储、管理和分析的数据集合。其核心特征通常概括为 **4V**：

| 特征 | 英文 | 含义 |
|------|------|------|
| **Volume** | 数据量 | 数据规模巨大，从TB到PB甚至EB级别 |
| **Velocity** | 速度 | 数据产生和处理的速度极快，要求实时或近实时响应 |
| **Variety** | 多样性 | 数据类型繁多，包括结构化、半结构化和非结构化数据 |
| **Value** | 价值 | 数据中蕴含高价值，但价值密度低，需要从海量数据中挖掘 |

除了4V之外，一些学者还补充了 **Veracity**（真实性/准确性）等维度。

大数据处理的核心思路：**移动计算比移动数据更划算**。将计算程序分发到数据所在的节点上执行，而非将数据集中传输到计算节点。

---

## 2. Hadoop架构概述

Hadoop 是 Apache 基金会开发的开源分布式计算框架，是大数据生态的基石。其核心由三个组件构成：

- **HDFS**（Hadoop Distributed File System）：分布式文件系统，负责海量数据的存储
- **MapReduce**：分布式计算模型，负责海量数据的并行处理
- **YARN**（Yet Another Resource Negotiator）：资源管理框架，负责集群资源的调度与管理

设计哲学：
- **硬件故障是常态**：系统应自动处理节点故障，而非依赖高可靠硬件
- **流式数据访问**：优化吞吐量而非延迟
- **大规模数据集**：支持GB到PB级别的数据
- **移动计算**：将计算迁移到数据所在位置
- **简单一致性模型**：采用"一次写入、多次读取"的模式

---

## 3. HDFS（Hadoop分布式文件系统）

### 3.1 NameNode与DataNode

HDFS 采用 **主从架构（Master-Slave）**：

- **NameNode（主节点）**：
  - 管理文件系统的命名空间（Namespace），维护文件系统树及树中所有文件和目录的元数据
  - 记录每个文件的数据块（Block）列表及其所在DataNode
  - 元数据全部存储在内存中，同时持久化到磁盘的 `fsimage`（镜像文件）和 `edits`（编辑日志）
  - 是HDFS的单点故障源（SPOF）

- **DataNode（从节点）**：
  - 负责实际数据块的存储和读写
  - 定期向NameNode发送心跳信号（Heartbeat）和块报告（Block Report）
  - 执行数据的创建、删除、复制等操作
  - 数据以文件形式存储在本地磁盘

- **Secondary NameNode**：
  - 并非NameNode的热备节点
  - 辅助NameNode合并 `fsimage` 和 `edits`，减轻NameNode的启动负担

### 3.2 块存储与副本机制

- **块（Block）**：HDFS将文件切分为固定大小的块，默认 **128MB**（Hadoop 2.x及以后，1.x为64MB）
  - 大块大小减少了元数据开销，适合大文件顺序读取
  - 每个块独立存储，最后一个块可能未满
  - 一个文件可以由多个块组成

- **副本机制（Replication）**：
  - 默认副本数为 **3**（可通过 `dfs.replication` 配置）
  - 副本放置策略（机架感知）：
    - 第1个副本：放在客户端所在的DataNode（如果客户端不在集群中，则随机选择）
    - 第2个副本：放在不同机架的DataNode上
    - 第3个副本：放在与第2个副本同一机架的不同节点上
  - 此策略在可靠性和写入带宽之间取得平衡

### 3.3 客户端读写流程

**读流程（Read）：**

1. 客户端向NameNode发起读请求，获取文件的块位置信息
2. NameNode返回每个块及其副本所在的DataNode列表（按距离排序）
3. 客户端选择最近的DataNode建立连接，读取数据块
4. 逐块读取，直到所有数据读完
5. 读取过程中若DataNode不可用，自动切换到其他副本节点

**写流程（Write）：**

1. 客户端向NameNode请求创建文件，NameNode进行权限和命名检查
2. 客户端将数据写入本地缓冲区，达到一个块大小时向NameNode请求分配DataNode
3. NameNode返回一组DataNode（默认3个），客户端与第一个DataNode建立管道（Pipeline）
4. 数据以Packet为单位沿Pipeline依次传输：客户端 → DN1 → DN2 → DN3
5. 每个DataNode接收数据后发送确认（ACK）沿Pipeline反向返回
6. 所有副本写入成功后，客户端通知NameNode关闭文件

### 3.4 高可用（HA）方案

Hadoop 2.x 引入了 NameNode 高可用方案：

- **Active/Standby 架构**：部署两个NameNode，一个为Active，一个为Standby
- **共享编辑日志**：使用 **QJM（Quorum Journal Manager）** 作为共享存储，Active NameNode将edits写入JournalNode集群（奇数个节点），Standby从JournalNode读取并同步
- **ZKFC（ZooKeeper Failover Controller）**：每个NameNode运行一个ZKFC进程，通过ZooKeeper进行故障检测和自动切换
- **DataNode 同时向两个NameNode发送块报告**
- **隔离机制（Fencing）**：确保同一时刻只有一个Active NameNode，通常通过SSH kill进程和撤销网络访问实现

Hadoop 3.x 还引入了 **多NameNode联邦（Federation）** 方案，允许一个集群有多个独立的NameNode，各自管理不同的命名空间。

---

## 4. MapReduce编程模型

### 4.1 Map阶段、Shuffle阶段、Reduce阶段

MapReduce 是一种 **分而治之** 的编程模型，将计算分为两个核心阶段，中间通过 Shuffle 阶段衔接：

- **Map阶段**：
  - 输入数据被切分为多个 **InputSplit**（通常与HDFS块对齐）
  - 每个Split由一个Map Task处理
  - Mapper以 `<key1, value1>` 为输入，输出 `<key2, value2>` 中间结果
  - 输出写入环形缓冲区（默认100MB），达到阈值（默认80%）时溢写（Spill）到磁盘

- **Shuffle阶段**（Map端）：
  - 环形缓冲区中数据按分区（Partition）排序，溢写到本地磁盘文件
  - 多个溢写文件最终合并（Merge）为一个有序文件
  - Combiner可在Map端预聚合，减少传输数据量

- **Shuffle阶段**（Reduce端）：
  - Reduce Task通过HTTP从各Map Task拉取（Fetch）属于自己分区的数据
  - 对拉取的数据进行归并排序（Merge）
  - 合并后的数据按key分组后传入Reducer

- **Reduce阶段**：
  - Reducer以 `<key2, list(value2)>` 为输入，输出最终结果 `<key3, value3>`
  - 输出写入HDFS

### 4.2 WordCount示例

```java
// Mapper
public class WordCountMapper extends Mapper<LongWritable, Text, Text, IntWritable> {
    private final static IntWritable one = new IntWritable(1);
    private Text word = new Text();

    @Override
    protected void map(LongWritable key, Text value, Context context)
            throws IOException, InterruptedException {
        String[] words = value.toString().split("\\s+");
        for (String w : words) {
            word.set(w);
            context.write(word, one);
        }
    }
}

// Reducer
public class WordCountReducer extends Reducer<Text, IntWritable, Text, IntWritable> {
    @Override
    protected void reduce(Text key, Iterable<IntWritable> values, Context context)
            throws IOException, InterruptedException {
        int sum = 0;
        for (IntWritable val : values) {
            sum += val.get();
        }
        context.write(key, new IntWritable(sum));
    }
}
```

### 4.3 Combiner与Partitioner

- **Combiner（合并器）**：
  - 在Map端进行局部聚合，本质是一个"本地Reducer"
  - 适用于满足结合律和交换律的操作（如求和、计数）
  - 不能用于求平均值等操作
  - 通过 `job.setCombinerClass()` 设置

- **Partitioner（分区器）**：
  - 决定Map输出的每条记录分配到哪个Reducer
  - 默认采用 HashPartitioner：`key.hashCode() % numReduceTasks`
  - 自定义Partitioner可实现按业务逻辑分区
  - 分区数应与Reduce Task数一致

### 4.4 MapReduce的局限性

- **执行模型简单**：只有Map和Reduce两个阶段，复杂任务需要多个MR Job串联
- **延迟高**：每个阶段都需要写磁盘，不适合迭代计算和交互式查询
- **不适合流式处理**：设计面向批处理，不支持实时数据流
- **开发效率低**：Java API繁琐，复杂逻辑代码量大
- **不适合DAG计算**：多个MR Job之间数据需要落盘，效率低

这些局限性催生了 Spark、Flink 等新一代计算框架。

---

## 5. YARN（资源管理）

### 5.1 ResourceManager与NodeManager

YARN 同样采用 **主从架构**：

- **ResourceManager（RM）**：
  - 全局资源管理器，负责整个集群的资源分配和调度
  - 包含两个核心组件：
    - **Scheduler（调度器）**：根据容量、队列等限制条件分配资源，不负责应用的监控和状态跟踪
    - **ApplicationsManager（ASM）**：管理所有Application，处理Job的提交和ApplicationMaster的启动

- **NodeManager（NM）**：
  - 每个节点上的代理，管理该节点上的资源
  - 向ResourceManager汇报本节点的资源使用情况（CPU、内存）
  - 接收ResourceManager的指令，管理Container的生命周期
  - 监控Container的资源使用并上报

### 5.2 ApplicationMaster

- 每个应用程序（Application）对应一个ApplicationMaster
- 负责向ResourceManager申请资源（Container）
- 与NodeManager通信，启动和监控任务
- 任务失败时负责重新申请资源并重启
- 运行在Container中，本身也占用资源
- 应用结束时自行注销

### 5.3 调度器

YARN 提供三种内置调度器：

- **FIFO Scheduler（先进先出）**：
  - 最简单的调度器，按提交顺序执行
  - 单队列，不支持并发
  - 大作业可能长时间阻塞后续作业
  - 适用于小规模测试环境

- **Capacity Scheduler（容量调度器）**：
  - 支持多队列，每个队列保证一定的资源容量
  - 队列内采用FIFO调度
  - 支持资源共享（空闲队列资源可借给其他队列）
  - 是Hadoop 2.x的默认调度器

- **Fair Scheduler（公平调度器）**：
  - 所有应用公平共享集群资源
  - 支持多队列，队列间公平分配
  - 队列内也支持公平调度或FIFO
  - 短作业可快速获得资源，长作业不会被完全饿死
  - 支持资源抢占（Preemption）

---

## 6. Hadoop生态组件

Hadoop 生态系统围绕核心三组件发展出丰富的工具链：

| 组件 | 功能 | 说明 |
|------|------|------|
| **Hive** | 数据仓库工具 | 提供类SQL查询语言（HQL），将SQL转换为MapReduce/Tez/Spark任务执行 |
| **HBase** | 分布式列式数据库 | 基于HDFS的NoSQL数据库，支持实时随机读写，适合海量稀疏数据 |
| **Pig** | 数据流处理平台 | 提供Pig Latin脚本语言，简化MapReduce编程 |
| **Sqoop** | 数据传输工具 | 在Hadoop与关系型数据库之间高效批量传输数据 |
| **Flume** | 日志采集系统 | 分布式、高可靠的日志采集、聚合和传输服务 |
| **Oozie** | 工作流调度引擎 | 管理和调度Hadoop作业（MapReduce、Hive、Pig等）的工作流 |

其他重要生态组件还包括：
- **ZooKeeper**：分布式协调服务，提供配置管理、命名服务、分布式锁、领导者选举
- **Tez**：基于DAG的计算框架，可替代MapReduce作为Hive的执行引擎
- **Ambari**：Hadoop集群的部署、管理和监控工具
- **Kafka**：分布式消息队列，常用于大数据管道中的数据传输
