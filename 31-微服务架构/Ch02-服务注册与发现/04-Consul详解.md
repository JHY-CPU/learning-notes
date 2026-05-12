# Consul 详解

## 一、Consul 核心功能

```
Consul 功能:
├── 服务发现 - HTTP/DNS 接口
├── 健康检查 - 多种检查方式
├── KV 存储 - 配置管理
├── 多数据中心 - 原生支持
└── Service Mesh - 内置 Connect
```

## 二、Spring Boot 集成

```yaml
spring:
  cloud:
    consul:
      host: localhost
      port: 8500
      discovery:
        service-name: user-service
        health-check-interval: 10s
        prefer-ip-address: true
```

## 三、Consul vs Nacos

| 维度 | Consul | Nacos |
|------|--------|-------|
| 语言 | Go | Java |
| 一致性 | CP | AP+CP |
| 配置 | 支持 | 支持 |
| 多数据中心 | 原生 | 不支持 |
| 国内生态 | 一般 | 好 |

## 四、Consul 工作原理

Consul 基于 Raft 协议实现强一致性，每个数据中心由 Server 节点组成 Raft 集群。服务注册时，客户端通过 HTTP API 或 DNS 将自身信息提交到 Agent，Agent 再转发给 Server 集群。健康检查由 Agent 执行，支持 HTTP、TCP、Script、gRPC 等多种方式，检查失败后自动从服务列表中移除。多数据中心之间通过 WAN Gossip 协议同步服务目录。

## 五、优缺点

**优点：**
- 原生多数据中心支持，适合跨地域部署
- 内置 Service Mesh（Consul Connect），支持 mTLS
- KV 存储可用于配置管理和分布式锁
- Go 编写，单二进制部署，资源占用低

**缺点：**
- CP 模式下网络分区时部分节点不可用
- 国内社区和文档资源相对较少
- 大规模集群下 Raft 同步有性能瓶颈

## 六、最佳实践

1. 生产环境至少部署 3 个 Server 节点
2. 健康检查间隔建议 10-30 秒，避免过于频繁
3. 使用 ACL 控制访问权限，保障安全性
4. 启用 Consul Connect 实现服务间加密通信

## 七、常见陷阱

1. **CP 模式下网络抖动可能导致服务短暂不可达**，需配合客户端缓存
2. **Agent 挂掉后服务不会自动注销**，需要设置 DeregisterCriticalServiceAfter
3. **KV 存储不适合存储大对象**，仅用于轻量配置
4. **多数据中心同步有延迟**，跨区调用需考虑一致性
