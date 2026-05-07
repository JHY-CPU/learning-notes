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

## 四、注意事项

1. **Consul 是 CP 模式**，强一致性
2. **多数据中心是核心优势**
3. **Go 语言性能好，资源占用低**
4. **国内使用不如 Nacos 广泛**
5. **Consul Connect 提供 Service Mesh 能力**
