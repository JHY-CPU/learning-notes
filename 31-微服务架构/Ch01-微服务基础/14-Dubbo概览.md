# Apache Dubbo 概览

## 一、Dubbo 简介

Apache Dubbo 是阿里巴巴开源的**高性能 RPC 框架**，专注于服务间高效通信。

```
Dubbo 核心功能:
├── 服务注册发现 - Nacos/Zookeeper
├── RPC 通信 - 高性能二进制协议
├── 负载均衡 - 多种策略
├── 熔断降级 - 内置支持
├── 序列化 - Hessian2/Protobuf
└── 服务治理 - 路由、权重、分组
```

## 二、基本使用

```java
// 服务提供者
@Service  // Dubbo 注解
public class UserServiceImpl implements UserService {
    @Override
    public User getUser(Long id) {
        return userMapper.selectById(id);
    }
}

// 服务消费者
@RestController
public class UserController {
    @Reference  // Dubbo 远程引用
    private UserService userService;

    @GetMapping("/users/{id}")
    public User getUser(@PathVariable Long id) {
        return userService.getUser(id);
    }
}
```

```yaml
# Dubbo 配置
dubbo:
  application:
    name: user-service
  registry:
    address: nacos://localhost:8848
  protocol:
    name: dubbo
    port: 20880
```

## 三、Dubbo vs Spring Cloud

| 维度 | Dubbo | Spring Cloud |
|------|-------|--------------|
| 协议 | Dubbo (TCP) | HTTP (REST) |
| 性能 | 更高 | 中等 |
| 生态 | 较小 | 更大 |
| 学习曲线 | 中等 | 较低 |
| 适用场景 | 高性能内部调用 | 通用微服务 |

## 四、Dubbo 3.0 新特性

```java
// Triple 协议 - 兼容 gRPC 和 HTTP
// proto 文件
syntax = "proto3";
service UserService {
    rpc GetUser(GetUserRequest) returns (User);
}

// Dubbo 3.0 使用 @DubboService 替代 @Service
@DubboService(version = "1.0.0", group = "user-group")
public class UserServiceImpl implements UserService {
    @Override
    public User getUser(GetUserRequest request) {
        return userRepository.findById(request.getId());
    }
}

// 消费端使用 @DubboReference
@RestController
public class UserController {
    @DubboService(version = "1.0.0", group = "user-group")
    private UserService userService;
}
```

```yaml
# application.yml - Dubbo 3.0 配置
dubbo:
  application:
    name: user-service
    qos-enable: true
    qos-port: 22222
  protocol:
    name: tri          # Triple 协议
    port: 50052
    serialization: protobuf
  registry:
    address: nacos://localhost:8848
    group: DEFAULT_GROUP
  consumer:
    check: false
    timeout: 3000
    retries: 2
```

## 五、Dubbo 高级特性

```java
// 泛化调用 - 不依赖接口 jar 包
ReferenceConfig<GenericService> reference = new ReferenceConfig<>();
reference.setInterface("com.example.UserService");
reference.setGeneric(true);
GenericService genericService = reference.get();
Object result = genericService.$invoke("getUser", new String[]{"java.lang.Long"}, new Object[]{1L});

// 异步调用
@DubboReference(async = true)
private UserService userService;

CompletableFuture<User> future = RpcContext.getContext()
    .getCompletableFuture();
```

## 六、Dubbo + Spring Cloud 混合架构

```
┌──────────────────────────────────────────┐
│              API Gateway                 │
│         (Spring Cloud Gateway)           │
└──────────┬───────────────┬───────────────┘
           │               │
    ┌──────▼──────┐ ┌──────▼──────┐
    │ Spring Cloud │ │   Dubbo     │
    │   服务集群   │ │  服务集群    │
    │  (REST)     │ │  (Triple)   │
    └─────────────┘ └─────────────┘
           │               │
    ┌──────▼───────────────▼──────┐
    │         Nacos 注册中心       │
    └─────────────────────────────┘
```

## 七、注意事项

1. **Dubbo 性能优于 REST** - TCP 长连接 + 二进制序列化，吞吐量高 5-10 倍
2. **Dubbo 3.0 支持 Triple 协议** - 基于 HTTP/2，兼容 gRPC 生态
3. **Spring Cloud 生态更完善** - 网关、配置中心、链路追踪组件更丰富
4. **国内公司 Dubbo 使用广泛** - 阿里、美团、滴滴等大量使用
5. **可以混合使用 Dubbo + Spring Cloud** - 内部调用用 Dubbo，对外用 REST
6. **关注 Dubbo 3.0 的应用级服务发现** - 性能比接口级发现好很多
