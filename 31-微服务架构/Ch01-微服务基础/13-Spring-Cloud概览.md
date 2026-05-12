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

## 三、Spring Cloud Alibaba 完整配置

```xml
<!-- Spring Cloud Alibaba 依赖 -->
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-discovery</artifactId>
</dependency>
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-nacos-config</artifactId>
</dependency>
<dependency>
    <groupId>com.alibaba.cloud</groupId>
    <artifactId>spring-cloud-starter-alibaba-sentinel</artifactId>
</dependency>
```

```yaml
# application.yml
spring:
  cloud:
    nacos:
      discovery:
        server-addr: nacos-server:8848
        namespace: dev
        group: DEFAULT_GROUP
    sentinel:
      transport:
        dashboard: sentinel-dashboard:8080
        port: 8719
```

## 四、Spring Cloud 与 Spring Boot 版本对应

| Spring Cloud | Spring Boot | 状态 |
|-------------|-------------|------|
| 2024.0.x | 3.4.x | 最新稳定 |
| 2023.0.x | 3.2.x / 3.3.x | LTS |
| 2022.0.x | 3.0.x / 3.1.x | 维护中 |
| 2021.0.x | 2.7.x | 即将 EOL |

## 五、典型项目结构

```
microservice-project/
├── pom.xml                        # 父 POM
├── api-gateway/                   # 网关服务
│   ├── src/main/java/
│   └── pom.xml
├── user-service/                  # 用户服务
│   ├── user-api/                  # 接口定义（Feign Client）
│   ├── user-core/                 # 核心实现
│   └── pom.xml
├── order-service/                 # 订单服务
│   ├── order-api/
│   ├── order-core/
│   └── pom.xml
├── common/                        # 公共模块
│   ├── common-core/               # 通用实体、工具
│   ├── common-security/           # 安全组件
│   └── common-redis/              # Redis 封装
└── docker-compose.yml             # 本地开发环境
```

```java
// OpenFeign 服务调用
@FeignClient(name = "user-service", fallbackFactory = UserClientFallback.class)
public interface UserClient {
    @GetMapping("/api/users/{id}")
    Result<UserDTO> getUser(@PathVariable("id") Long id);

    @PostMapping("/api/users/batch")
    Result<List<UserDTO>> batchGetUsers(@RequestBody List<Long> ids);
}

// 熔断降级
@Component
public class UserClientFallback implements FallbackFactory<UserClient> {
    @Override
    public UserClient create(Throwable cause) {
        return new UserClient() {
            @Override
            public Result<UserDTO> getUser(Long id) {
                return Result.success(UserDTO.defaultUser(id));
            }
            @Override
            public Result<List<UserDTO>> batchGetUsers(List<Long> ids) {
                return Result.success(Collections.emptyList());
            }
        };
    }
}
```

## 六、注意事项

1. **Spring Cloud 版本要与 Spring Boot 版本匹配** - 参照官方版本对应表
2. **Nacos 替代了 Eureka + Config** - 一个组件同时提供注册发现和配置管理
3. **Sentinel 替代了 Hystrix** - Hystrix 已停更，Sentinel 是更好的选择
4. **Spring Cloud Alibaba 在国内更流行** - 社区活跃，中文文档完善
5. **选择组件时要考虑社区活跃度** - 避免选择已停更的组件
6. **生产环境做好容灾** - 网关、注册中心都需要高可用部署
