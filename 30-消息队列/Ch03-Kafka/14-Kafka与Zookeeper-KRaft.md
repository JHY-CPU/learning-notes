# Kafka 与 ZooKeeper / KRaft

## 一、ZooKeeper 模式（传统架构）

Kafka 0.x-3.x 依赖 ZooKeeper 存储集群元数据、选举 Controller、管理 Broker 注册。

```
ZooKeeper 集群:
  /brokers/ids/1          -> Broker 1 信息
  /brokers/ids/2          -> Broker 2 信息
  /brokers/topics/xxx     -> Topic 元数据
  /controller             -> Controller 信息
  /consumers/xxx/offsets  -> 消费者 Offset（旧版）

Kafka Brokers:
  Controller (Broker1) <-> Broker2 <-> Broker3
```

```properties
# ZooKeeper 配置 (旧版 server.properties)
zookeeper.connect=zk1:2181,zk2:2181,zk3:2181
zookeeper.connection.timeout.ms=18000
zookeeper.session.timeout.ms=18000
```

### ZooKeeper 模式的缺点

```
- 两套系统需要运维（Kafka + ZooKeeper）
- ZooKeeper 是元数据瓶颈（分区数上限约 20 万）
- Controller 切换需要从 ZK 重建状态（慢）
- ZK 不适合存储大量数据
```

## 二、KRaft 模式（新架构）

Kafka 2.8 引入 KRaft 模式（Kafka Raft），用内置 Raft 协议替代 ZooKeeper，3.0+ 生产就绪，3.5+ 默认启用。

```
KRaft 集群:
  Controller (Broker1) --Raft-- Controller (Broker2) --Raft-- Controller (Broker3)
       │                          │                              │
       └──────────────────────────┼──────────────────────────────┘
                                  │
                          __cluster_metadata Topic
                          (元数据存储在 Kafka 内部)
```

```properties
# KRaft 模式配置 (3.5+ server.properties)
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
controller.listener.names=CONTROLLER

listeners=PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
advertised.listeners=PLAINTEXT://kafka1:9092
```

### KRaft 优势

```
优势                    说明
----------------------------------------------------------
单一系统                不再依赖 ZooKeeper
更快的启动              不需要从 ZK 加载元数据
更快的故障恢复          Raft 协议快速选举
更高的分区支持          支持百万级分区
更简单的运维            只需管理 Kafka 集群
```

## 三、从 ZooKeeper 迁移到 KRaft

```bash
# 1. 生成迁移存储
kafka-storage.sh format \
  --cluster-id MkU3OEVBNTcwNTJENDM2Qk \
  --config kraft/server.properties

# 2. 启动 KRaft Controller（先启动 3 个 Controller）
kafka-server-start.sh kraft-controller.properties

# 3. 逐个将 Broker 迁移到 KRaft
# 停止 Broker -> 修改配置 -> 使用 kafka-metadata-shell 验证 -> 启动

# 4. 验证迁移
kafka-metadata-shell --snapshot /path/to/__cluster_metadata-0/00000000000000000000.log
>> ls /brokers
>> ls /topics
```

## 四、KRaft 内部机制

```
__cluster_metadata Topic:
  Partition 0:
    Offset 0: [RegisterBroker] Broker 1 注册
    Offset 1: [RegisterBroker] Broker 2 注册
    Offset 2: [RegisterBroker] Broker 3 注册
    Offset 3: [CreateTopic] order-events
    Offset 4: [PartitionChange] order-events partition 0 leader=1

Controller Quorum:
  Leader Controller -> 处理元数据变更
  Follower Controller -> 复制元数据日志
```

## 五、注意事项

1. **KRaft 是 Kafka 的未来方向**，新集群务必使用 KRaft 模式
2. **迁移过程中 Topic 不会中断**，可以在线完成
3. **Controller 节点建议 3 个或 5 个**，保证容灾
4. **KRaft 模式支持更多分区**（百万级），ZooKeeper 模式约 20 万上限
5. **3.5+ 版本 KRaft 是默认模式**，不再需要任何 ZooKeeper 配置
