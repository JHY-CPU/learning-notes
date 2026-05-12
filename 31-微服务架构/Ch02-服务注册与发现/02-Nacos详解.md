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

## 四、Nacos 集群部署

```yaml
# cluster.conf (每个节点配置)
192.168.1.10:8848
192.168.1.11:8848
192.168.1.12:8848
```

```bash
# Docker Compose 集群
docker-compose -f cluster-mode.yaml up -d

# cluster-mode.yaml
version: '3.8'
services:
  nacos1:
    image: nacos/nacos-server:v2.3.0
    environment:
      - MODE=cluster
      - NACOS_SERVERS=nacos1:8848 nacos2:8848 nacos3:8848
      - MYSQL_SERVICE_HOST=mysql
      - MYSQL_SERVICE_DB_NAME=nacos
      - MYSQL_SERVICE_USER=nacos
      - MYSQL_SERVICE_PASSWORD=nacos
    ports: ["8848:8848", "9848:9848", "9849:9849"]

  nacos2:
    image: nacos/nacos-server:v2.3.0
    environment:
      - MODE=cluster
      - NACOS_SERVERS=nacos1:8848 nacos2:8848 nacos3:8848
    ports: ["8849:8848", "9850:9848", "9851:9849"]

  nacos3:
    image: nacos/nacos-server:v2.3.0
    environment:
      - MODE=cluster
      - NACOS_SERVERS=nacos1:8848 nacos2:8848 nacos3:8848
    ports: ["8850:8848", "9852:9848", "9853:9849"]

  mysql:
    image: mysql:8.0
    environment:
      - MYSQL_ROOT_PASSWORD=root
      - MYSQL_DATABASE=nacos
```

## 五、Nacos 权重与灰度

```java
// 通过元数据实现灰度发布
spring:
  cloud:
    nacos:
      discovery:
        metadata:
          version: v2          # 版本号
          region: east          # 区域标签
          weight: "100"         # 权重

// 消费端按版本路由
@Component
public class VersionRule extends AbstractLoadBalancerRule {
    @Override
    public Server choose(Object key) {
        List<Server> servers = getLoadBalancer().getReachableServers();
        // 只选择版本匹配的实例
        return servers.stream()
            .filter(s -> "v2".equals(s.getMetaInfo().get("version")))
            .findFirst()
            .orElse(servers.get(0));
    }
}
```

## 六、Nacos 与 Eureka 对比

| 特性 | Nacos | Eureka |
|------|-------|--------|
| 一致性 | AP+CP | AP |
| 配置管理 | 内置 | 需要 Config Server |
| 健康检查 | TCP/HTTP/MySQL | 心跳 |
| 管理界面 | 功能丰富 | 基础 |
| 社区状态 | 阿里活跃维护 | Netflix 停止维护 |
| 雪崩保护 | 支持 | 自我保护模式 |

## 七、注意事项

1. **Nacos 2.0+ 使用 gRPC** - 端口需要额外开放 9848(gRPC) 和 9849(Raft)
2. **命名空间用于环境隔离** - dev/test/prod 使用不同命名空间
3. **临时实例用 AP 模式** - 适合无状态服务；持久实例用 CP 模式
4. **配置变更会实时推送** - 使用长轮询 + gRPC 推送
5. **生产环境至少 3 节点集群** - 搭配 MySQL 存储保证数据可靠
6. **注意 Nacos 1.x 升级 2.x** - 通信协议从 HTTP 改为 gRPC，端口配置不同
