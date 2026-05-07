# APM应用性能

## 一、概念说明

APM（Application Performance Monitoring）用于监控应用程序的性能，包括响应时间、吞吐量、错误率、数据库查询等。常见工具有SkyWalking、Pinpoint、Datadog APM等。

| 工具 | 特点 | 适用场景 |
|------|------|----------|
| SkyWalking | 开源、Java强、插件丰富 | Java微服务 |
| Pinpoint | 开源、低侵入 | Java应用 |
| Datadog | 商业、全栈 | 多语言 |
| New Relic | 商业、易用 | Web应用 |

## 二、具体用法

### SkyWalking部署

```yaml
# docker-compose.yml
version: '3.8'
services:
  oap:
    image: apache/skywalking-oap-server:9.7.0
    environment:
      SW_STORAGE: elasticsearch
      SW_STORAGE_ES_CLUSTER_NODES: elasticsearch:9200
    ports:
      - 11800:11800  # gRPC
      - 12800:12800  # HTTP

  ui:
    image: apache/skywalking-ui:9.7.0
    environment:
      SW_OAP_ADDRESS: http://oap:12800
    ports:
      - 8080:8080
```

### Java Agent集成

```bash
# 启动时添加Agent
java -javaagent:/path/to/skywalking-agent.jar \
     -Dskywalking.agent.service_name=my-java-app \
     -Dskywalking.collector.backend_service=oap:11800 \
     -jar my-app.jar
```

```yaml
# agent.config
agent.service_name=${SW_AGENT_NAME:my-java-app}
collector.backend_service=${SW_AGENT_COLLECTOR_BACKEND_SERVICES:oap:11800}
agent.sample_n_per_3_secs=${SW_AGENT_SAMPLE:1}
logging.level=${SW_LOG_LEVEL:INFO}
```

### Python Agent

```python
# pip install skywalking

from skywalking import agent, config

config.init(
    agent_name='my-python-app',
    agent_instance_name='instance-1',
    collector_address='oap:11800',
    protocol='grpc',
)
agent.start()

# 自动追踪Flask、Django、FastAPI等框架
from flask import Flask
app = Flask(__name__)

# 自定义追踪
from skywalking import trace, Component

@app.route('/api/data')
def get_data():
    with trace.create_local_span(op='custom_operation') as span:
        span.component = Component.Flask
        span.tag('key', 'value')
        # 业务逻辑
        return process()

# 跨线程追踪
from skywalking import trace_context

def background_task():
    carrier = trace_context.capture()
    def run():
        trace_context.continued(carrier)
        with trace.create_local_span(op='background'):
            pass
    thread = threading.Thread(target=run)
    thread.start()
```

### 性能指标分析

```bash
# 关键APM指标
服务指标:
  - 服务响应时间 (P50, P95, P99)
  - 吞吐量 (TPM/RPM)
  - 错误率
  - SLA达标率

实例指标:
  - CPU/内存使用率
  - GC频率和时间
  - 线程池状态
  - 连接池状态

端点指标:
  - 接口响应时间
  - 接口调用次数
  - 接口错误率
  - 慢接口列表

追踪数据:
  - Trace瀑布图
  - Span耗时分布
  - 调用链拓扑图
  - 错误Trace详情
```

### 告警配置

```yaml
# SkyWalking告警规则
rules:
  - name: service_resp_time_rule
    expression: service_resp_time > 1000
    period: 10
    message: "服务 {name} 响应时间超过1秒，当前: {value}ms"
    tags:
      level: WARNING

  - name: service_sla_rule
    expression: service_sla < 9500
    period: 10
    message: "服务 {name} SLA低于95%，当前: {value}"
    tags:
      level: CRITICAL

  - name: service_resp_time_percentile_rule
    expression: service_resp_time_percentile{p='95'} > 2000
    period: 10
    message: "服务 {name} P95延迟超过2秒"
    tags:
      level: WARNING
```

## 三、注意事项与常见陷阱

1. **Agent开销**：Agent有性能开销，评估对应用的影响
2. **数据量控制**：设置合理的采样率
3. **存储规划**：APM数据量大，需规划存储容量
4. **跨语言支持**：确认Agent支持使用的编程语言
5. **版本兼容**：Agent版本与框架版本要兼容
6. **安全传输**：Agent与OAP之间使用TLS加密
7. **性能基线**：建立性能基线便于异常检测
