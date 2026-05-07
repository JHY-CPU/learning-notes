# Eureka 详解

## 一、Eureka 架构

```
Eureka Server (注册中心)
├── 服务注册表
├── 自我保护模式
└── 多节点复制

Eureka Client
├── 服务提供者 - 注册
├── 服务消费者 - 发现
└── 心跳续约 - 默认 30 秒
```

## 二、Eureka Server

```java
@SpringBootApplication
@EnableEurekaServer
public class EurekaServerApplication { }
```

```yaml
# Eureka Server 配置
eureka:
  client:
    register-with-eureka: false
    fetch-registry: false
  server:
    enable-self-preservation: true  # 自我保护
    eviction-interval-timer-in-ms: 60000
```

## 三、Eureka Client

```java
@SpringBootApplication
@EnableDiscoveryClient
public class UserServiceApplication { }
```

```yaml
eureka:
  client:
    service-url:
      defaultZone: http://localhost:8761/eureka/
  instance:
    prefer-ip-address: true
    lease-renewal-interval-in-seconds: 30
```

## 四、注意事项

1. **Eureka 是 AP 模式**，优先可用性
2. **自我保护模式防止网络分区误删服务**
3. **Netflix 已停止维护 Eureka**，建议用 Nacos
4. **多节点 Eureka 相互注册**
5. **客户端缓存服务列表**，Eureka 挂了仍可工作
