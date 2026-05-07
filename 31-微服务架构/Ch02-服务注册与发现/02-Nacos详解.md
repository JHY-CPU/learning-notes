# Nacos 详解

## 一、Nacos 核心功能

```
Nacos 功能:
├── 服务注册发现 - 服务自动注册和发现
├── 配置管理 - 动态配置推送
├── 服务元数据 - 权重、集群、健康状态
└── 命名空间 - 多环境隔离
```

## 二、Spring Boot 集成

```yaml
# application.yml
spring:
  cloud:
    nacos:
      discovery:
        server-addr: localhost:8848
        namespace: dev
        group: DEFAULT_GROUP
        cluster-name: default
      config:
        server-addr: localhost:8848
        file-extension: yaml
        shared-configs:
          - data-id: common.yaml
            group: DEFAULT_GROUP
```

## 三、服务注册与发现

```java
// 自动注册（EnableDiscoveryClient）
@SpringBootApplication
@EnableDiscoveryClient
public class OrderServiceApplication { }

// 编程式服务发现
@Autowired
private DiscoveryClient discoveryClient;

public List<ServiceInstance> getInstances() {
    return discoveryClient.getInstances("order-service");
}
```

## 四、注意事项

1. **Nacos 2.0+ 使用 gRPC**，端口需要额外开放 9848/9849
2. **命名空间用于环境隔离**（dev/test/prod）
3. **临时实例用 AP 模式**，持久实例用 CP 模式
4. **配置变更会实时推送**，无需重启服务
5. **生产环境至少 3 节点集群**
