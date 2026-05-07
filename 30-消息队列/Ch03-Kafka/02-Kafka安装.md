# Kafka 安装

## 一、Docker Compose 安装（KRaft 模式）

```yaml
# docker-compose.yml - Kafka 3.7+ KRaft 模式（无 ZooKeeper）
version: '3.8'
services:
  kafka:
    image: confluentinc/cp-kafka:7.6.0
    container_name: kafka
    ports:
      - "9092:9092"    # 客户端通信
      - "9093:9093"    # Controller 通信
    environment:
      # KRaft 模式配置
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka:9093
      KAFKA_CONTROLLER_LISTENER_NAMES: CONTROLLER
      # Broker 配置
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_LISTENER_SECURITY_PROTOCOL_MAP: PLAINTEXT:PLAINTEXT,CONTROLLER:PLAINTEXT
      KAFKA_INTER_BROKER_LISTENER_NAME: PLAINTEXT
      # 集群 ID
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk
      # 其他配置
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_REPLICATION_FACTOR: 1
      KAFKA_TRANSACTION_STATE_LOG_MIN_ISR: 1
      KAFKA_LOG_DIRS: /tmp/kraft-combined-logs
    volumes:
      - kafka_data:/tmp/kraft-combined-logs

  # Kafka UI（可选）
  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka-ui
    ports:
      - "8080:8080"
    environment:
      KAFKA_CLUSTERS_0_NAME: local
      KAFKA_CLUSTERS_0_BOOTSTRAPSERVERS: kafka:9092

volumes:
  kafka_data:
```

```bash
# 启动
docker-compose up -d

# 验证
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list
```

## 二、Docker 集群部署

```yaml
# docker-compose-cluster.yml
version: '3.8'
services:
  kafka1:
    image: confluentinc/cp-kafka:7.6.0
    hostname: kafka1
    environment:
      KAFKA_NODE_ID: 1
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka1:9092
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk
    ports:
      - "9092:9092"

  kafka2:
    image: confluentinc/cp-kafka:7.6.0
    hostname: kafka2
    environment:
      KAFKA_NODE_ID: 2
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka2:9092
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk
    ports:
      - "9093:9092"

  kafka3:
    image: confluentinc/cp-kafka:7.6.0
    hostname: kafka3
    environment:
      KAFKA_NODE_ID: 3
      KAFKA_PROCESS_ROLES: broker,controller
      KAFKA_CONTROLLER_QUORUM_VOTERS: 1@kafka1:9093,2@kafka2:9093,3@kafka3:9093
      KAFKA_LISTENERS: PLAINTEXT://0.0.0.0:9092,CONTROLLER://0.0.0.0:9093
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka3:9092
      CLUSTER_ID: MkU3OEVBNTcwNTJENDM2Qk
    ports:
      - "9094:9092"
```

## 三、常用命令

```bash
# Topic 操作
kafka-topics --bootstrap-server localhost:9092 --list
kafka-topics --bootstrap-server localhost:9092 --create \
  --topic test-topic --partitions 6 --replication-factor 1
kafka-topics --bootstrap-server localhost:9092 --describe --topic test-topic
kafka-topics --bootstrap-server localhost:9092 --delete --topic test-topic

# 生产消息
kafka-console-producer --bootstrap-server localhost:9092 --topic test-topic

# 消费消息
kafka-console-consumer --bootstrap-server localhost:9092 \
  --topic test-topic --from-beginning --group test-group

# 消费者组
kafka-consumer-groups --bootstrap-server localhost:9092 --list
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group test-group

# 重置 Offset
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group test-group --topic test-topic --reset-offsets --to-earliest --execute
```

## 四、Java 客户端连接

```java
// 生产者
Properties producerProps = new Properties();
producerProps.put("bootstrap.servers", "localhost:9092");
producerProps.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
producerProps.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");
producerProps.put("acks", "all");

KafkaProducer<String, String> producer = new KafkaProducer<>(producerProps);
producer.send(new ProducerRecord<>("test-topic", "key", "value"));
producer.close();

// 消费者
Properties consumerProps = new Properties();
consumerProps.put("bootstrap.servers", "localhost:9092");
consumerProps.put("group.id", "test-group");
consumerProps.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
consumerProps.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
consumerProps.put("auto.offset.reset", "earliest");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(consumerProps);
consumer.subscribe(List.of("test-topic"));
while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(100));
    for (ConsumerRecord<String, String> record : records) {
        System.out.printf("key=%s, value=%s%n", record.key(), record.value());
    }
}
```

## 五、注意事项

1. **`advertised.listeners` 必须配置为客户端能访问的地址**，否则连接不上
2. **集群部署时 `replication-factor` 不能超过 Broker 数量**
3. **KRaft 模式不需要 ZooKeeper**，3.0+ 版本推荐使用
4. **数据目录要挂载到高速 SSD**，Kafka 对磁盘 I/O 要求很高
5. **`server.properties` 中 `broker.id` 必须全局唯一**
