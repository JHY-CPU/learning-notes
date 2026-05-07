# 监控 Kafka

## 一、JMX 监控

Kafka 通过 JMX 暴露丰富的运行时指标。

```bash
# 启动时开启 JMX
export JMX_PORT=9999
kafka-server-start.sh config/server.properties

# 或 Docker 环境
KAFKA_JMX_PORT=9999
KAFKA_JMX_HOSTNAME=localhost
```

## 二、核心监控指标

```yaml
Broker 指标:
  - kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec
    每秒消息数（吞吐量）
  - kafka.server:type=BrokerTopicMetrics,name=BytesInPerSec
    每秒写入字节
  - kafka.server:type=BrokerTopicMetrics,name=BytesOutPerSec
    每秒读取字节
  - kafka.controller:type=KafkaController,name=ActiveControllerCount
    活跃 Controller 数量（应为 1）
  - kafka.server:type=ReplicaManager,name=UnderReplicatedPartitions
    副本不足的分区数（应为 0）

Topic 指标:
  - kafka.server:type=BrokerTopicMetrics,name=TotalProduceRequestsPerSec,topic=xxx
    指定 Topic 生产速率
  - kafka.server:type=BrokerTopicMetrics,name=TotalFetchRequestsPerSec,topic=xxx
    指定 Topic 消费速率

消费者组指标:
  - kafka.consumer:type=consumer-fetch-manager-metrics,client-id=xxx,topic=xxx
    消费速率、延迟
```

## 三、Prometheus + Grafana 监控

```yaml
# docker-compose-monitoring.yml
version: '3.8'
services:
  kafka-exporter:
    image: danielqsj/kafka-exporter:latest
    command:
      - --kafka.server=kafka:9092
    ports:
      - "9308:9308"

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
```

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka-exporter:9308']
```

## 四、关键告警规则

```yaml
# prometheus-alerts.yml
groups:
  - name: kafka
    rules:
      - alert: KafkaUnderReplicatedPartitions
        expr: kafka_server_replicamanager_underreplicatedpartitions > 0
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Kafka 存在副本不足的分区"

      - alert: KafkaConsumerLagHigh
        expr: kafka_consumergroup_lag > 100000
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "消费者组 {{ $labels.consumergroup }} 消费延迟过高"

      - alert: KafkaBrokerDown
        expr: up{job="kafka"} == 0
        for: 1m
        labels:
          severity: critical

      - alert: KafkaDiskUsageHigh
        expr: kafka_log_size_bytes / disk_total_bytes > 0.8
        for: 5m
        labels:
          severity: warning
```

## 五、命令行监控

```bash
# 查看消费者组延迟
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --describe --group my-group

# 查看 Topic 详情
kafka-topics --bootstrap-server localhost:9092 \
  --describe --topic order-events

# 查看 Broker 状态
kafka-metadata --snapshot /tmp/kraft-combined-logs/__cluster_metadata-0/00000000000000000000.log \
  --cluster-id MkU3OEVBNTcwNTJENDM2Qk
```

## 六、注意事项

1. **UnderReplicatedPartitions > 0 是严重告警**，说明有副本落后
2. **消费延迟（LAG）是最核心的消费者监控指标**
3. **ActiveControllerCount 应该始终为 1**，多个 Controller 说明有脑裂
4. **监控磁盘使用率**，Kafka 数据目录满了会导致服务不可用
5. **JMX 监控有性能开销**，建议 30 秒采集一次
