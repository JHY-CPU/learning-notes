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

## 四、Eureka 高可用集群

```yaml
# eureka-server-1.yml
server:
  port: 8761
eureka:
  instance:
    hostname: eureka-server-1
  client:
    register-with-eureka: true
    fetch-registry: true
    service-url:
      defaultZone: http://eureka-server-2:8762/eureka/,http://eureka-server-3:8763/eureka/

---
# eureka-server-2.yml
server:
  port: 8762
eureka:
  instance:
    hostname: eureka-server-2
  client:
    register-with-eureka: true
    fetch-registry: true
    service-url:
      defaultZone: http://eureka-server-1:8761/eureka/,http://eureka-server-3:8763/eureka/
```

## 五、自我保护机制详解

```
自我保护触发条件:
  最近 1 分钟内续约比例 < 85%（默认阈值）

自我保护行为:
├── 停止服务剔除 - 保留所有注册实例
├── 正常接收注册和续约
└── 日志输出进入自我保护模式

退出条件:
├── 续约比例恢复正常（> 85%）
└── 网络恢复后自动退出

注意: 自我保护可能导致调用到已宕机的服务
     需要配合客户端重试和熔断使用
```

```yaml
# 自我保护配置
eureka:
  server:
    enable-self-preservation: true        # 生产环境必须开启
    renewal-percent-threshold: 0.85       # 续约比例阈值
    eviction-interval-timer-in-ms: 60000  # 剔除间隔
```

## 六、Eureka 源码要点

```java
// 服务注册核心流程
// 1. PeerAwareInstanceRegistryImpl.register()
//    → 将实例信息存入 ConcurrentHashMap
//    → 同步到其他 Eureka 节点

// 2. 定时任务 EvictionTask
//    → 扫描超过 90 秒未续约的实例
//    → 非自我保护模式下剔除

// 3. 客户端缓存刷新
//    → 默认每 30 秒全量拉取注册表
//    → 增量拉取每 30 秒
```

## 七、注意事项

1. **Eureka 是 AP 模式** - 优先可用性，允许短暂数据不一致
2. **自我保护模式防止网络分区误删服务** - 生产环境必须开启
3. **Netflix 已停止维护 Eureka** - 建议迁移到 Nacos
4. **多节点 Eureka 相互注册** - 至少 3 个节点形成集群
5. **客户端缓存服务列表** - Eureka 挂了仍可工作，提供容灾能力
6. **迁移到 Nacos 的注意事项** - Nacos 支持 AP/CP 双模式，功能更丰富
