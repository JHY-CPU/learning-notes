# Spring Cloud 概览

## 一、Spring Cloud 生态

```
Spring Cloud 全家桶:
├── Spring Cloud Gateway    - API 网关
├── Nacos                   - 注册中心 + 配置中心
├── OpenFeign               - 声明式 HTTP 客户端
├── LoadBalancer            - 负载均衡
├── Sentinel                - 熔断降级
├── SkyWalking/Zipkin       - 链路追踪
├── Spring Cloud Bus        - 消息总线
└── Spring Cloud Stream     - 消息驱动
```

```xml
<!-- Spring Cloud 依赖 -->
<dependencyManagement>
    <dependencies>
        <dependency>
            <groupId>org.springframework.cloud</groupId>
            <artifactId>spring-cloud-dependencies</artifactId>
            <version>2023.0.0</version>
            <type>pom</type>
            <scope>import</scope>
        </dependency>
    </dependencies>
</dependencyManagement>
```

## 二、快速开始

```java
// 服务注册
@SpringBootApplication
@EnableDiscoveryClient
public class UserServiceApplication { }

// 服务调用
@FeignClient(name = "order-service")
public interface OrderClient {
    @GetMapping("/orders/{id}")
    Order getOrder(@PathVariable Long id);
}
```

## 三、注意事项

1. **Spring Cloud 版本要与 Spring Boot 版本匹配**
2. **Nacos 替代了 Eureka + Config**
3. **Sentinel 替代了 Hystrix**
4. **Spring Cloud Alibaba 在国内更流行**
5. **选择组件时要考虑社区活跃度**
