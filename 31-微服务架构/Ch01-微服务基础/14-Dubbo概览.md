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

## 四、注意事项

1. **Dubbo 性能优于 REST**，适合内部服务调用
2. **Dubbo 3.0 支持 Triple 协议**（基于 HTTP/2）
3. **Spring Cloud 生态更完善**
4. **国内公司 Dubbo 使用广泛**
5. **可以混合使用 Dubbo + Spring Cloud**
