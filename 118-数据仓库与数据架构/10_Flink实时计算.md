# Apache Flink 实时计算

## Flink架构概览

Apache Flink是一个分布式流处理框架，提供高吞吐、低延迟、精确一次(Exactly-Once)语义的有状态计算能力。

```text
+----------------------------------------------+
|               Flink Applications              |
|   DataStream API | Table API | SQL | ML       |
+----------------------------------------------+
|              Runtime (分布式数据流)             |
|  TaskManager  |  TaskManager  |  TaskManager  |
+----------------------------------------------+
|              Resource Management              |
|           YARN / K8s / Standalone             |
+----------------------------------------------+
```

## DataStream API（Java）

### 基础词频统计

```java
import org.apache.flink.api.common.functions.FlatMapFunction;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.util.Collector;

public class WordCount {
    public static void main(String[] args) throws Exception {
        // 1. 创建执行环境
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        env.setParallelism(2);

        // 2. 读取数据源
        DataStream<String> text = env.socketTextStream("localhost", 9999);

        // 3. 转换处理
        DataStream<Tuple2<String, Integer>> counts = text
            .flatMap(new FlatMapFunction<String, Tuple2<String, Integer>>() {
                @Override
                public void flatMap(String value, Collector<Tuple2<String, Integer>> out) {
                    for (String word : value.toLowerCase().split("\\s+")) {
                        if (word.length() > 0) {
                            out.collect(new Tuple2<>(word, 1));
                        }
                    }
                }
            })
            .keyBy(value -> value.f0)
            .sum(1);

        // 4. 输出结果
        counts.print();

        // 5. 执行
        env.execute("Streaming WordCount");
    }
}
```

### 数据源与Sink

```java
import org.apache.flink.api.common.serialization.SimpleStringSchema;
import org.apache.flink.connector.kafka.source.KafkaSource;
import org.apache.flink.connector.kafka.source.enumerator.initializer.OffsetsInitializer;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

public class KafkaToFlink {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

        // Kafka Source
        KafkaSource<String> source = KafkaSource.<String>builder()
            .setBootstrapServers("kafka-broker1:9092,kafka-broker2:9092")
            .setTopics("user-events")
            .setGroupId("flink-consumer-group")
            .setStartingOffsets(OffsetsInitializer.latest())
            .setValueOnlyDeserializer(new SimpleStringSchema())
            .build();

        // Kafka Sink
        KafkaSink<String> sink = KafkaSink.<String>builder()
            .setBootstrapServers("kafka-broker1:9092")
            .setRecordSerializer(KafkaRecordSerializationSchema.builder()
                .setTopic("processed-events")
                .setValueSerializationSchema(new SimpleStringSchema())
                .build())
            .build();

        env.fromSource(source, WatermarkStrategy.noWatermarks(), "Kafka Source")
           .map(value -> {
               // 数据处理逻辑
               return value.toUpperCase();
           })
           .sinkTo(sink);

        env.execute("Kafka to Flink Pipeline");
    }
}
```

## 窗口操作（Windowing）

窗口是流处理的核心概念，将无界数据流切分为有限的数据块进行计算。

### 时间语义

```java
import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.common.eventtime.SerializableTimestampAssigner;
import java.time.Duration;

// 定义Watermark策略（处理乱序数据，允许5秒延迟）
WatermarkStrategy<Event> watermarkStrategy = WatermarkStrategy
    .<Event>forBoundedOutOfOrderness(Duration.ofSeconds(5))
    .withTimestampAssigner(new SerializableTimestampAssigner<Event>() {
        @Override
        public long extractTimestamp(Event element, long recordTimestamp) {
            return element.getEventTime();  // 从事件数据中提取时间戳
        }
    })
    .withIdleness(Duration.ofMinutes(1));  // 处理空闲分区

DataStream<Event> withTimestampsAndWatermarks = events
    .assignTimestampsAndWatermarks(watermarkStrategy);
```

### 滚动窗口(Tumbling Window)

```java
// 每5分钟统计一次用户点击量
events
    .assignTimestampsAndWatermarks(watermarkStrategy)
    .keyBy(Event::getUserId)
    .window(TumblingEventTimeWindows.of(Time.minutes(5)))
    .aggregate(new CountAggregate())
    .addSink(new JdbcSink(...));
```

### 滑动窗口(Sliding Window)

```java
// 每1分钟统计过去10分钟的UV
events
    .assignTimestampsAndWatermarks(watermarkStrategy)
    .keyBy(Event::getPageId)
    .window(SlidingEventTimeWindows.of(Time.minutes(10), Time.minutes(1)))
    .aggregate(new UvAggregate());
```

### 会话窗口(Session Window)

```java
// 会话间隔超过30分钟则开启新会话
events
    .assignTimestampsAndWatermarks(watermarkStrategy)
    .keyBy(Event::getUserId)
    .window(EventTimeSessionWindows.withGap(Time.minutes(30)))
    .process(new SessionProcessFunction());
```

## 状态管理

Flink的状态管理是其实现有状态流处理的核心能力。

### Keyed State

```java
import org.apache.flink.api.common.state.ValueState;
import org.apache.flink.api.common.state.ValueStateDescriptor;
import org.apache.flink.api.common.state.MapState;
import org.apache.flink.api.common.state.MapStateDescriptor;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.KeyedProcessFunction;
import org.apache.flink.util.Collector;

public class FraudDetector extends KeyedProcessFunction<String, Transaction, Alert> {

    // 声明状态
    private ValueState<Boolean> flagState;
    private ValueState<Long> timerState;
    private MapState<String, Double> amountState;

    @Override
    public void open(Configuration parameters) {
        // 初始化状态
        flagState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("flag", Boolean.class));
        timerState = getRuntimeContext().getState(
            new ValueStateDescriptor<>("timer-state", Long.class));
        amountState = getRuntimeContext().getMapState(
            new MapStateDescriptor<>("amount-by-type", String.class, Double.class));
    }

    @Override
    public void processElement(Transaction txn, Context ctx, Collector<Alert> out) throws Exception {
        // 读取状态
        Boolean lastFlag = flagState.value();

        // 累计各类型交易金额
        Double currentAmount = amountState.get(txn.getType());
        amountState.put(txn.getType(), (currentAmount == null ? 0 : currentAmount) + txn.getAmount());

        // 欺诈检测逻辑：小金额交易后紧跟大金额交易
        if (lastFlag != null && txn.getAmount() > 5000) {
            out.collect(new Alert(txn.getUserId(), "疑似欺诈交易"));
            cleanUp(ctx);
        }

        // 标记小金额交易
        if (txn.getAmount() < 1.0) {
            flagState.update(true);
            long timer = ctx.timerService().currentProcessingTime() + 60000; // 1分钟窗口
            ctx.timerService().registerProcessingTimeTimer(timer);
            timerState.update(timer);
        }
    }

    @Override
    public void onTimer(long timestamp, OnTimerContext ctx, Collector<Alert> out) {
        timerState.clear();
        flagState.clear();
    }

    private void cleanUp(Context ctx) throws Exception {
        Long timer = timerState.value();
        if (timer != null) {
            ctx.timerService().deleteProcessingTimeTimer(timer);
        }
        timerState.clear();
        flagState.clear();
    }
}
```

### Operator State

```java
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.streaming.api.functions.sink.SinkFunction;
import org.apache.flink.streaming.api.checkpoint.ListCheckpointed;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;

public class BufferingSink implements SinkFunction<Tuple2<String, Long>>,
        ListCheckpointed<Tuple2<String, Long>> {

    private final int batchSize;
    private List<Tuple2<String, Long>> bufferedElements;

    public BufferingSink(int batchSize) {
        this.batchSize = batchSize;
        this.bufferedElements = new ArrayList<>();
    }

    @Override
    public void invoke(Tuple2<String, Long> value, Context context) {
        bufferedElements.add(value);
        if (bufferedElements.size() >= batchSize) {
            // 批量写入外部系统
            flush();
        }
    }

    @Override
    public List<Tuple2<String, Long>> snapshotState(long checkpointId, long timestamp) {
        return new ArrayList<>(bufferedElements);
    }

    @Override
    public void restoreState(List<Tuple2<String, Long>> state) {
        bufferedElements.addAll(state);
    }

    private void flush() {
        // 实际写入逻辑（如写入数据库）
        System.out.println("Flushing " + bufferedElements.size() + " records");
        bufferedElements.clear();
    }
}
```

## Flink SQL

### 流式SQL查询

```sql
-- 定义Kafka Source Table
CREATE TABLE user_events (
    user_id     BIGINT,
    event_type  STRING,
    page_url    STRING,
    event_time  TIMESTAMP(3),
    WATERMARK FOR event_time AS event_time - INTERVAL '5' SECOND
) WITH (
    'connector' = 'kafka',
    'topic' = 'user-events',
    'properties.bootstrap.servers' = 'kafka:9092',
    'properties.group.id' = 'flink-sql-group',
    'scan.startup.mode' = 'latest-offset',
    'format' = 'json'
);

-- 定义MySQL Sink Table
CREATE TABLE user_metrics (
    user_id       BIGINT,
    window_start  TIMESTAMP(3),
    window_end    TIMESTAMP(3),
    click_count   BIGINT,
    event_types   STRING,
    PRIMARY KEY (user_id, window_start) NOT ENFORCED
) WITH (
    'connector' = 'jdbc',
    'url' = 'jdbc:mysql://mysql:3306/analytics',
    'table-name' = 'user_metrics',
    'username' = 'root',
    'password' = 'password'
);

-- 流式聚合：每5分钟统计用户行为
INSERT INTO user_metrics
SELECT
    user_id,
    window_start,
    window_end,
    COUNT(*) AS click_count,
    STRING_AGG(DISTINCT event_type, ',') AS event_types
FROM TABLE(
    TUMBLE(TABLE user_events, DESCRIPTOR(event_time), INTERVAL '5' MINUTES)
)
GROUP BY user_id, window_start, window_end;

-- CEP模式匹配：检测连续登录失败
SELECT *
FROM user_events
MATCH_RECOGNIZE (
    PARTITION BY user_id
    ORDER BY event_time
    MEASURES
        A.event_time AS start_time,
        LAST(B.event_time) AS fail_time,
        COUNT(*) AS fail_count
    ONE ROW PER MATCH
    AFTER MATCH SKIP PAST LAST ROW
    PATTERN (A B{3,})   -- 连续3次及以上
    DEFINE
        A AS A.event_type = 'login_fail',
        B AS B.event_type = 'login_fail'
);
```

## Exactly-Once语义

Flink通过Checkpoint机制实现端到端的精确一次语义。

### Checkpoint配置

```java
import org.apache.flink.streaming.api.CheckpointingMode;
import org.apache.flink.streaming.api.environment.CheckpointConfig;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;

StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();

// 启用Checkpoint（每30秒一次）
env.enableCheckpointing(30000);
env.getCheckpointConfig().setCheckpointingMode(CheckpointingMode.EXACTLY_ONCE);
env.getCheckpointConfig().setMinPauseBetweenCheckpoints(10000);
env.getCheckpointConfig().setCheckpointTimeout(60000);
env.getCheckpointConfig().setMaxConcurrentCheckpoints(1);
env.getCheckpointConfig().setTolerableCheckpointFailureNumber(3);

// Checkpoint失败时取消任务
env.getCheckpointConfig().setExternalizedCheckpointCleanup(
    CheckpointConfig.ExternalizedCheckpointCleanup.RETAIN_ON_CANCELLATION);

// 状态后端配置
env.setStateBackend(new HashMapStateBackend());
env.getCheckpointConfig().setCheckpointStorage("hdfs://namenode:8020/flink/checkpoints");
```

### 二阶段提交Sink（Kafka Exactly-Once）

```java
import org.apache.flink.connector.kafka.sink.KafkaSink;
import org.apache.flink.connector.kafka.sink.KafkaRecordSerializationSchema;
import org.apache.flink.connector.base.DeliveryGuarantee;

KafkaSink<String> sink = KafkaSink.<String>builder()
    .setBootstrapServers("kafka:9092")
    .setRecordSerializer(KafkaRecordSerializationSchema.builder()
        .setTopic("output-topic")
        .setValueSerializationSchema(new SimpleStringSchema())
        .build())
    .setDeliveryGuarantee(DeliveryGuarantee.EXACTLY_ONCE)  // 精确一次
    .setTransactionalIdPrefix("flink-kafka-sink")           // 事务ID前缀
    .setProperty("transaction.timeout.ms", "900000")        // 事务超时
    .build();
```

## Python API (PyFlink)

### 基础流处理

```python
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.window import TumblingProcessingTimeWindows
from pyflink.datastream.functions import MapFunction, ReduceFunction
from pyflink.common import WatermarkStrategy, Time
from pyflink.common.typeinfo import Types

env = StreamExecutionEnvironment.get_execution_environment()
env.set_parallelism(2)

# 数据源
ds = env.from_collection(
    collection=[("user_1", 100), ("user_2", 200), ("user_1", 150), ("user_3", 300)],
    type_info=Types.TUPLE([Types.STRING(), Types.INT()])
)

# 窗口聚合
result = ds \
    .key_by(lambda x: x[0]) \
    .window(TumblingProcessingTimeWindows.of(Time.seconds(10))) \
    .reduce(lambda a, b: (a[0], a[1] + b[1]))

result.print()
env.execute("PyFlink Window Example")
```

### Flink SQL with Python

```python
from pyflink.table import EnvironmentSettings, TableEnvironment

env_settings = EnvironmentSettings.in_streaming_mode()
t_env = TableEnvironment.create(env_settings)

# 定义Kafka Source
t_env.execute_sql("""
    CREATE TABLE orders (
        order_id BIGINT,
        user_id BIGINT,
        amount DECIMAL(10, 2),
        order_time TIMESTAMP(3),
        WATERMARK FOR order_time AS order_time - INTERVAL '5' SECOND
    ) WITH (
        'connector' = 'kafka',
        'topic' = 'orders',
        'properties.bootstrap.servers' = 'localhost:9092',
        'scan.startup.mode' = 'latest-offset',
        'format' = 'json'
    )
""")

# 流式查询
result = t_env.execute_sql("""
    SELECT
        user_id,
        TUMBLE_START(order_time, INTERVAL '1' MINUTE) AS window_start,
        SUM(amount) AS total_amount
    FROM orders
    GROUP BY
        TUMBLE(order_time, INTERVAL '1' MINUTE),
        user_id
""")

# 打印结果
with result.collect() as results:
    for row in results:
        print(f"User: {row[0]}, Window: {row[1]}, Total: {row[2]}")
```

## 最佳实践

1. **合理设置并行度**：根据数据量和CPU核心数设置，一般等于Kafka分区数
2. **状态后端选择**：大状态使用RocksDBStateBackend，小状态使用HashMapStateBackend
3. **Checkpoint间隔**：根据业务容忍度设置，通常10秒-5分钟
4. **Watermark策略**：根据数据乱序程度设置容忍延迟，避免数据丢失
5. **背压处理**：监控背压指标，必要时增加并行度或优化算子逻辑
6. **算子链控制**：合理设置算子链共享策略，避免不必要的数据序列化
