# HDFS与MapReduce


## HDFS与MapReduce


大数据Hadoop分布式存储


Hadoop分布式文件系统(HDFS)与MapReduce编程范式是大数据生态的基石。


## HDFS架构


HDFS（Hadoop Distributed File System）采用主从架构（Master-Slave），专为大规模数据集设计。


```
核心组件：
┌─────────────────────────────────────────────┐
│              HDFS 架构                       │
│                                             │
│  NameNode (主节点)                           │
│  ├── 管理文件系统命名空间 (Namespace)          │
│  ├── 维护文件到数据块的映射                    │
│  ├── 记录每个数据块的位置信息                  │
│  └── 处理客户端的文件操作请求                  │
│                                             │
│  DataNode (从节点)                           │
│  ├── 存储实际的数据块 (Block)                 │
│  ├── 执行数据的读写操作                       │
│  ├── 向NameNode汇报心跳和块报告               │
│  └── 执行数据的复制和校验                     │
└─────────────────────────────────────────────┘
```


> **Note:** HDFS默认块大小为128MB（Hadoop 2.x+），远大于普通文件系统的4KB，减少寻址开销。


## NameNode与DataNode详解


```
NameNode 元数据管理：
- FsImage：文件系统元数据的完整快照
- EditLog：记录所有文件系统的变更操作
- Secondary NameNode：定期合并FsImage和EditLog（非热备！）

DataNode 数据管理：
- 数据块默认3副本（可配置）
- 副本放置策略：机架感知
  第1副本：客户端所在节点
  第2副本：不同机架的节点
  第3副本：与第2副本同机架的不同节点
- 定期向NameNode发送心跳（默认3秒）

高可用方案（HDFS HA）：
- Active/Standby NameNode
- 使用ZooKeeper进行故障切换
- 共享EditLog（QJM：Quorum Journal Manager）
```


## MapReduce编程范式


MapReduce是一种分布式计算框架，将计算分为Map和Reduce两个阶段。


```
MapReduce 执行流程：
Input → Split → Map → Shuffle → Reduce → Output

1. Input Split：将输入数据切分为多个分片
2. Map阶段：对每个分片并行执行Map函数
   输入：<key1, value1>
   输出：<key2, value2>（中间结果）
3. Shuffle阶段（核心）：
   - Partition：按Key分区
   - Sort：对Key排序
   - Combiner：本地预聚合（可选）
   - Copy：从Map节点拉取数据到Reduce节点
4. Reduce阶段：对每个Key的值列表进行聚合
   输入：<key2, [value2, ...]>
   输出：<key3, value3>
5. Output：将结果写入HDFS
```


> **Note:** WordCount是MapReduce的经典示例：Map将每行拆分为单词(word,1)，Reduce对每个单词的计数求和。


## YARN资源管理


YARN（Yet Another Resource Negotiator）是Hadoop的资源管理层，将资源管理和任务调度分离。


```
YARN 架构：
┌──────────────────────────────────────────┐
│  ResourceManager (全局资源管理器)          │
│  ├── Scheduler：资源调度（FIFO/Capacity/Fair）│
│  └── ApplicationMaster：管理应用生命周期    │
│                                          │
│  NodeManager (节点管理器)                  │
│  ├── 管理单个节点的资源                     │
│  ├── 启动和监控Container                   │
│  └── 向ResourceManager汇报资源使用          │
│                                          │
│  ApplicationMaster                        │
│  ├── 每个应用一个AM                        │
│  ├── 向ResourceManager申请资源              │
│  └── 与NodeManager通信启动任务              │
│                                          │
│  Container（资源容器）                      │
│  ├── 封装某个节点的CPU/内存资源              │
│  └── 任务在Container中执行                 │
└──────────────────────────────────────────┘
```


> **Note:** YARN支持多种计算框架：MapReduce、Spark、Flink、Tez等，实现了计算框架与资源管理的解耦。


## HDFS读写流程


```
写入流程 (HDFS Write)：
1. 客户端向NameNode请求写入文件
2. NameNode检查权限和文件是否存在，返回可写入的DataNode列表
3. 客户端将数据写入第一个DataNode（Pipeline方式）
4. 数据以Packet为单位在Pipeline中传输
5. 每个DataNode收到数据后写入本地磁盘并转发给下一个
6. 确认信息沿Pipeline反向传回客户端
7. 客户端关闭文件，NameNode提交元数据

读取流程 (HDFS Read)：
1. 客户端向NameNode请求读取文件
2. NameNode返回数据块的位置信息（按距离排序）
3. 客户端选择最近的DataNode直接读取数据
4. 读取完一个块后，校验数据完整性
5. 如果读取失败，尝试读取副本
6. 关闭连接
```


<!-- Converted from: 01_HDFS与MapReduce.html -->
