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

## 三、注意事项

1. **Zipkin 简单轻量**
2. **Sleuth 自动传播 TraceID**
3. **采样率要根据 QPS 设置**
4. **与 Kafka 集成可异步上报**
5. **功能不如 SkyWalking 全面**
