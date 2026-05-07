# 链路追踪Jaeger

## 一、概念说明

Jaeger是Uber开源的分布式追踪系统，兼容OpenTracing/OpenTelemetry标准。用于监控和排查微服务架构中的请求链路问题。

| 概念 | 说明 |
|------|------|
| Trace | 一个完整请求的调用链 |
| Span | 链路中的一个操作单元 |
| SpanContext | 跨服务传递的上下文 |
| Baggage | 跨Span传递的键值对数据 |

## 二、具体用法

### 部署Jaeger

```yaml
# docker-compose.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:1.52
    environment:
      - COLLECTOR_OTLP_ENABLED=true
    ports:
      - 16686:16686  # UI
      - 14268:14268  # HTTP Collector
      - 6831:6831/udp  # UDP Agent
      - 4317:4317  # OTLP gRPC
      - 4318:4318  # OTLP HTTP
    volumes:
      - jaeger-data:/badger
```

### Python集成（OpenTelemetry）

```python
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from flask import Flask, request

# 配置追踪
resource = Resource.create({"service.name": "order-service"})
provider = TracerProvider(resource=resource)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer(__name__)
app = Flask(__name__)

@app.route('/order', methods=['POST'])
def create_order():
    with tracer.start_as_current_span("create_order") as span:
        # 设置Span属性
        order_data = request.json
        span.set_attribute("order.id", order_data["id"])
        span.set_attribute("order.amount", order_data["amount"])

        # 调用库存服务
        with tracer.start_as_current_span("check_inventory") as child_span:
            inventory = check_inventory(order_data["items"])
            child_span.set_attribute("inventory.available", inventory["available"])

        # 调用支付服务
        with tracer.start_as_current_span("process_payment") as child_span:
            payment = process_payment(order_data["amount"])
            child_span.set_attribute("payment.status", payment["status"])

        return {"order_id": order_data["id"], "status": "created"}
```

### Java Spring Boot集成

```java
// application.yml
management:
  tracing:
    sampling:
      probability: 1.0

// pom.xml依赖
// <dependency>
//   <groupId>io.micrometer</groupId>
//   <artifactId>micrometer-tracing-bridge-otel</artifactId>
// </dependency>
// <dependency>
//   <groupId>io.opentelemetry</groupId>
//   <artifactId>opentelemetry-exporter-otlp</artifactId>
// </dependency>

@Service
public class OrderService {
    private final Tracer tracer;

    public Order createOrder(OrderRequest request) {
        Span span = tracer.nextSpan().name("createOrder").start();
        try (Tracer.SpanInScope ws = tracer.withSpan(span)) {
            span.tag("order.amount", String.valueOf(request.getAmount()));
            // 业务逻辑
            return orderRepository.save(order);
        } finally {
            span.end();
        }
    }
}
```

### Node.js集成

```javascript
const { NodeSDK } = require('@opentelemetry/sdk-node');
const { JaegerExporter } = require('@opentelemetry/exporter-jaeger');

const sdk = new NodeSDK({
    serviceName: 'api-gateway',
    traceExporter: new JaegerExporter({
        endpoint: 'http://jaeger:14268/api/traces',
    }),
});
sdk.start();

const opentelemetry = require('@opentelemetry/api');
const tracer = opentelemetry.trace.getTracer('api-gateway');

app.get('/api/users', async (req, res) => {
    const span = tracer.startSpan('get_users');
    try {
        const users = await db.query('SELECT * FROM users');
        span.setAttribute('user.count', users.length);
        res.json(users);
    } finally {
        span.end();
    }
});
```

### 查询和分析

```bash
# Jaeger UI访问
http://localhost:16686

# 通过API查询
curl "http://jaeger:16666/api/traces?service=order-service&limit=20"

# 查询特定Trace
curl "http://jaeger:16666/api/traces/{traceID}"

# 查询服务列表
curl "http://jaeger:16666/api/services"
```

## 三、注意事项与常见陷阱

1. **采样策略**：生产环境使用概率采样（1-10%）避免性能影响
2. **存储选择**：开发用内存存储，生产用Cassandra/Elasticsearch
3. **上下文传播**：确保跨服务调用时传播Trace上下文
4. **异常记录**：捕获异常时记录到Span中
5. **Span命名**：使用有意义的名称，避免动态值
6. **资源消耗**：追踪有性能开销，生产环境需评估
7. **敏感信息**：避免在Span属性中记录密码等敏感数据
