# Kafka Streams

## 一、什么是 Kafka Streams

Kafka Streams 是一个**客户端库**，用于构建实时流处理应用。它直接嵌入到 Java 应用中，无需额外的集群。

```java
// 核心概念
// Stream: 无界的、连续的数据记录序列
// Table: 变更日志的当前状态视图
// KStream: 消息流
// KTable: 变更日志表
```

## 二、Hello World 示例

```java
public class WordCountApp {
    public static void main(String[] args) {
        Properties props = new Properties();
        props.put(StreamsConfig.APPLICATION_ID_CONFIG, "wordcount-app");
        props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
        props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

        StreamsBuilder builder = new StreamsBuilder();

        // 从输入 Topic 读取
        KStream<String, String> textLines = builder.stream("text-input");

        // 处理: 分词 -> 计数 -> 输出
        KTable<String, Long> wordCounts = textLines
            .flatMapValues(line -> Arrays.asList(line.toLowerCase().split("\\W+")))
            .groupBy((key, word) -> word)
            .count(Materialized.as("word-counts-store"));

        // 输出到 Topic
        wordCounts.toStream().to("word-count-output",
            Produced.with(Serdes.String(), Serdes.Long()));

        KafkaStreams streams = new KafkaStreams(builder.build(), props);
        streams.start();
    }
}
```

## 三、流处理操作

```java
StreamsBuilder builder = new StreamsBuilder();
KStream<String, String> source = builder.stream("input-topic");

// 1. 转换操作
KStream<String, String> upper = source.mapValues(value -> value.toUpperCase());

// 2. 过滤操作
KStream<String, String> filtered = source.filter((key, value) -> value.length() > 10);

// 3. 分组聚合
KTable<String, Long> counts = source
    .groupByKey()
    .count();

// 4. 窗口聚合
KTable<Windowed<String>, Long> windowedCounts = source
    .groupByKey()
    .windowedBy(TimeWindows.ofSizeWithNoGrace(Duration.ofMinutes(5)))
    .count();

// 5. 流-表 Join
KStream<String, String> enriched = ordersStream.join(
    usersTable,
    (order, user) -> order + " by " + user.getName()
);

// 6. 流-流 Join
KStream<String, String> joined = orderStream.join(
    paymentStream,
    (order, payment) -> order + " paid by " + payment,
    JoinWindows.ofTimeDifferenceWithNoGrace(Duration.ofMinutes(10))
);

// 输出
upper.to("output-topic");
```

## 四、状态存储

```java
// 内置状态存储（RocksDB）
StreamsBuilder builder = new StreamsBuilder();

// 创建全局状态表
GlobalKTable<String, String> globalTable = builder.globalTable("user-profiles");

// 在处理器中访问状态
KStream<String, String> processed = source.process(
    () -> new Processor<String, String, String, String>() {
        private KeyValueStore<String, String> store;

        @Override
        public void init(ProcessorContext<String, String> context) {
            store = context.getStateStore("my-store");
        }

        @Override
        public void process(Record<String, String> record) {
            String previous = store.get(record.key());
            store.put(record.key(), record.value());
            context.forward(record);
        }
    },
    "my-store"
);
```

## 五、注意事项

1. **APPLICATION_ID 要全局唯一**，用于区分不同的 Streams 应用
2. **状态存储默认用 RocksDB**，数据持久化到磁盘
3. **窗口操作需要处理延迟数据**，设置合理的 grace period
4. **多实例部署自动分区并行处理**
5. **Kafka Streams 不需要额外集群**，直接嵌入应用即可
