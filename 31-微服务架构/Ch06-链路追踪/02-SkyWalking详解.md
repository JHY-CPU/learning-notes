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

## 四、注意事项

1. **Agent 是无侵入的**，不需要改代码
2. **采样率要根据 QPS 调整**
3. **存储选择 ES 性能更好**
4. **SkyWalking 支持多种语言**
5. **生产环境 OAP 至少 2 节点**
