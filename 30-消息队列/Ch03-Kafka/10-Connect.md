# Kafka Connect

## 一、什么是 Kafka Connect

Kafka Connect 是一个**数据集成框架**，用于在 Kafka 和外部系统（数据库、搜索引擎、文件系统）之间可靠地传输数据。

```
数据源 (Source) --> Kafka (Topic) --> 数据汇 (Sink)
   MySQL            orders-topic          Elasticsearch
   MongoDB          users-topic           S3
   文件系统         logs-topic            Redis
```

## 二、Source Connector（数据导入）

```json
// MySQL Source Connector 配置
{
  "name": "mysql-source-connector",
  "config": {
    "connector.class": "io.debezium.connector.mysql.MySqlConnector",
    "tasks.max": "1",
    "database.hostname": "localhost",
    "database.port": "3306",
    "database.user": "debezium",
    "database.password": "dbz",
    "database.server.id": "184054",
    "database.include.list": "mydb",
    "table.include.list": "mydb.orders,mydb.users",
    "topic.prefix": "mysql",
    "schema.history.internal.kafka.bootstrap.servers": "localhost:9092",
    "schema.history.internal.kafka.topic": "schema-changes"
  }
}
```

```bash
# 提交 Connector
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @mysql-source-connector.json
```

## 三、Sink Connector（数据导出）

```json
// Elasticsearch Sink Connector 配置
{
  "name": "es-sink-connector",
  "config": {
    "connector.class": "io.confluent.connect.elasticsearch.ElasticsearchSinkConnector",
    "tasks.max": "3",
    "topics": "orders",
    "connection.url": "http://localhost:9200",
    "type.name": "_doc",
    "key.ignore": "false",
    "schema.ignore": "true",
    "behavior.on.null.values": "delete",
    "behavior.on.malformed.documents": "warn"
  }
}
```

## 四、常用 Connectors

```yaml
常用 Source Connectors:
  - Debezium (MySQL/PostgreSQL/MongoDB CDC)
  - JDBC Source (定时轮询)
  - S3 Source
  - File Source

常用 Sink Connectors:
  - Elasticsearch Sink
  - JDBC Sink
  - S3 Sink
  - Redis Sink
  - MongoDB Sink
```

## 五、Connect REST API

```bash
# 查看所有 Connector
curl http://localhost:8083/connectors

# 查看 Connector 状态
curl http://localhost:8083/connectors/mysql-source/status

# 暂停 Connector
curl -X PUT http://localhost:8083/connectors/mysql-source/pause

# 恢复 Connector
curl -X PUT http://localhost:8083/connectors/mysql-source/resume

# 删除 Connector
curl -X DELETE http://localhost:8083/connectors/mysql-source

# 查看 Connector 配置
curl http://localhost:8083/connectors/mysql-source/config

# 更新配置
curl -X PUT http://localhost:8083/connectors/mysql-source/config \
  -H "Content-Type: application/json" \
  -d '{"connector.class":"...","tasks.max":"2"}'
```

## 六、自定义 Connector

```java
// 自定义 Source Connector
public class CustomSourceConnector extends SourceConnector {
    @Override
    public void start(Map<String, String> props) {
        // 初始化配置
    }

    @Override
    public Class<? extends Task> taskClass() {
        return CustomSourceTask.class;
    }

    @Override
    public List<Map<String, String>> taskConfigs(int maxTasks) {
        // 返回每个 Task 的配置
    }
}

// 自定义 Source Task
public class CustomSourceTask extends SourceTask {
    @Override
    public List<SourceRecord> poll() {
        // 从外部系统读取数据
        List<SourceRecord> records = new ArrayList<>();
        // 转换为 SourceRecord
        records.add(new SourceRecord(
            sourcePartition,
            sourceOffset,
            "topic",
            keySchema, key,
            valueSchema, value
        ));
        return records;
    }
}
```

## 七、注意事项

1. **Connect Worker 可以集群部署**，自动分配 Task
2. **转换器（Converter）要与数据格式匹配**，JSON/Avro/Protobuf
3. **Debezium CDC 比 JDBC Source 性能好很多**，推荐用于数据库同步
4. **Sink Connector 的偏移量管理由框架处理**，无需手动管理
5. **生产环境至少 3 个 Worker 节点**，保证高可用
