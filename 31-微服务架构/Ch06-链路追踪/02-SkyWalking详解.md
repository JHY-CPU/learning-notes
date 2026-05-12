# SkyWalking 详解

## 一、架构

```
SkyWalking 架构:
├── Agent - Java Agent，无侵入采集
├── OAP Server - 数据分析和存储
├── Storage - 存储（ES/MySQL/H2）
└── UI - Web 可视化界面
```

## 二、Spring Boot 集成

```bash
# 启动时加载 Agent
java -javaagent:/path/skywalking-agent.jar \
     -Dskywalking.agent.service_name=user-service \
     -Dskywalking.collector.backend_service=oap:11800 \
     -jar user-service.jar
```

```yaml
# 配置
skywalking:
  agent:
    service_name: user-service
    sample_n_per_3_secs: 10  # 采样率
  collector:
    backend_service: oap:11800
```

## 三、核心功能

| 功能 | 说明 |
|------|------|
| 链路追踪 | 全链路调用跟踪 |
| 性能分析 | 耗时 TopN、慢查询 |
| 服务拓扑 | 自动发现依赖关系 |
| 告警 | 自定义告警规则 |
| 日志关联 | TraceID 关联日志 |

## 四、SkyWalking 集群部署

```yaml
# docker-compose.yml - SkyWalking 完整部署
version: '3.8'
services:
  oap:
    image: apache/skywalking-oap-server:9.4.0
    environment:
      SW_STORAGE: elasticsearch
      SW_STORAGE_ES_CLUSTER_NODES: es-node1:9200,es-node2:9200
      SW_CLUSTER: standalone
      SW_CORE_DATA_KEEPER_EXECUTE_PERIOD: 5
    ports:
      - "11800:11800"   # gRPC
      - "12800:12800"   # HTTP

  ui:
    image: apache/skywalking-ui:9.4.0
    environment:
      SW_OAP_ADDRESS: http://oap:12800
    ports:
      - "8080:8080"

  elasticsearch:
    image: elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - "ES_JAVA_OPTS=-Xms1g -Xmx1g"
    ports:
      - "9200:9200"
```

## 五、日志关联 TraceID

```xml
<!-- logback-spring.xml - 日志输出 TraceID -->
<configuration>
    <appender name="CONSOLE" class="ch.qos.logback.core.ConsoleAppender">
        <encoder>
            <pattern>%d{yyyy-MM-dd HH:mm:ss.SSS} [%thread] [%X{tid}] %-5level %logger{36} - %msg%n</pattern>
        </encoder>
    </appender>
</configuration>
```

```java
// 自定义日志关联 TraceID
@Component
public class TraceLoggingInterceptor implements HandlerInterceptor {
    @Override
    public boolean preHandle(HttpServletRequest request, HttpServletResponse response, Object handler) {
        String traceId = request.getHeader("sw8");  // SkyWalking Header
        if (traceId != null) {
            MDC.put("tid", traceId);
        }
        return true;
    }

    @Override
    public void afterCompletion(HttpServletRequest request, HttpServletResponse response, Object handler, Exception ex) {
        MDC.clear();
    }
}
```

## 六、SkyWalking 告警规则

```yaml
# alarm-settings.yml
rules:
  - name: service_resp_time_rule
    expression: service_resp_time > 1000
    period: 10
    count: 3
    silence-period: 5
    message: "服务 {name} 响应时间超过 1 秒，当前值 {value}ms"

  - name: service_sla_rule
    expression: service_sla < 95
    period: 10
    count: 2
    message: "服务 {name} SLA 低于 95%，当前值 {value}%"

  - name: service_p99_rule
    expression: service_p99 > 2000
    period: 10
    count: 3
    message: "服务 {name} P99 延迟超过 2 秒"

webhooks:
  - url: http://alert-manager:8080/api/alert
```

## 七、注意事项

1. **Agent 是无侵入的** - 通过 Java Agent 字节码增强，不需要改代码
2. **采样率要根据 QPS 调整** - QPS > 1000 的服务建议采样率 10-30%
3. **存储选择 ES 性能更好** - 生产环境用 ES 集群，H2/MySQL 仅限测试
4. **SkyWalking 支持多种语言** - Java/Go/Python/Node.js/.NET/PHP
5. **生产环境 OAP 至少 2 节点** - 避免单点故障
6. **Agent 与部分字节码框架冲突** - 遇到问题检查兼容性列表
