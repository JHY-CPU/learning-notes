# Zipkin 详解

## 一、Zipkin 架构

```
Zipkin 架构:
├── Collector - 收集链路数据
├── Storage - 存储（ES/MySQL/内存）
├── API - 查询接口
└── UI - Web 界面
```

## 二、Spring Boot 集成

```yaml
spring:
  zipkin:
    base-url: http://localhost:9411
    sender:
      type: web
  sleuth:
    sampler:
      probability: 1.0  # 采样率 100%
```

## 三、工作原理

Zipkin 基于 Google Dapper 论文实现分布式链路追踪。每次请求生成全局唯一的 TraceID，经过每个服务时创建一个 Span，Span 包含 SpanID、ParentSpanID、服务名、操作名、时间戳、耗时等信息。Sleuth 在 Spring Cloud 中自动拦截 HTTP 调用、MQ 消息、定时任务等，在请求头中传递 TraceID 和 SpanID。Zipkin Collector 接收客户端上报的 Span 数据，存入 ES 或 MySQL，API 层提供查询接口，UI 展示调用链路瀑布图和依赖关系图。

## 四、优缺点

**优点：**
- 轻量简单，部署和使用门槛低
- 支持多种存储后端（ES、MySQL、Cassandra、内存）
- 可通过 Kafka 异步上报，降低对业务的影响

**缺点：**
- 功能相对基础，缺少指标聚合和拓扑图
- 需要代码埋点或依赖 Sleuth 自动集成
- UI 功能简单，不如 SkyWalking 丰富

## 五、最佳实践

1. 生产环境采样率设为 0.1-0.5，高 QPS 场景降低采样率
2. 使用 Kafka 作为上报通道，解耦业务和追踪系统
3. 存储选择 ES，支持全文检索和聚合查询
4. 配置数据保留策略，定期清理旧数据

## 六、Zipkin + Kafka 上报

```yaml
# 通过 Kafka 异步上报 Span 数据，降低对业务性能影响
spring:
  zipkin:
    base-url: http://localhost:9411
    sender:
      type: kafka           # 改为 Kafka 上报
    kafka:
      topic: zipkin
  sleuth:
    sampler:
      probability: 0.3       # 30% 采样率
```

```yaml
# Zipkin Server 使用 Kafka 收集
# docker-compose.yml
version: '3.8'
services:
  zipkin:
    image: openzipkin/zipkin:3
    environment:
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
      - STORAGE_TYPE=elasticsearch
      - ES_HOSTS=http://elasticsearch:9200
    ports:
      - "9411:9411"
```

## 七、Brave 手动埋点

```java
// Brave 手动创建 Span - 更精细的追踪
@Autowired
private Tracer tracer;

public Order processOrder(Long orderId) {
    Span span = tracer.nextSpan().name("processOrder").start();
    try (Tracer.SpanInScope ws = tracer.withSpanInScope(span)) {
        span.tag("order.id", String.valueOf(orderId));

        // 子操作
        Span childSpan = tracer.nextSpan().name("validateOrder").start();
        try (Tracer.SpanInScope ws2 = tracer.withSpanInScope(childSpan)) {
            validateOrder(orderId);
        } finally {
            childSpan.finish();
        }

        return orderRepository.findById(orderId);
    } catch (Exception e) {
        span.error(e);
        throw e;
    } finally {
        span.finish();
    }
}
```

## 八、常见陷阱

1. **采样率 100% 在高 QPS 下影响性能** - 且存储压力巨大，生产建议 10-30%
2. **TraceID 在异步线程中丢失** - 需要手动传递 MDC 或使用 Brave 的 Tracer
3. **Kafka 上报通道故障导致 Span 数据丢失** - 需监控队列积压
4. **ES 存储不清理数据** - 磁盘空间耗尽，设置 ILM 策略
5. **Spring Cloud Sleuth 已不再维护** - 迁移到 Micrometer Tracing + OpenTelemetry
