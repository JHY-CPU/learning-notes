# APISIX 网关

## 一、APISIX 简介

Apache APISIX 是云原生、高性能的 API 网关，基于 Nginx + Lua 开发。

```bash
# Docker 启动 APISIX
docker run -d --name apisix \
  -p 9080:9080 \
  -p 9443:9443 \
  apache/apisix:latest
```

## 二、核心特性

```
特性:
├── 动态路由 - 热更新，无需重启
├── 丰富插件 - 80+ 插件
├── 多协议 - HTTP/gRPC/TCP/UDP
├── 可观测 - Prometheus/SkyWalking
└── 云原生 - K8s 原生支持
```

## 三、注意事项

1. **APISIX 国内社区活跃**
2. **动态路由是核心优势**
3. **性能与 Kong 相当**
4. **Dashboard 提供可视化管理**
5. **适合云原生环境**
