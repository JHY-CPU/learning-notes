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

## 三、工作原理

APISIX 同样基于 Nginx + LuaJIT，但使用 etcd 作为配置中心存储路由和插件配置，通过 etcd 的 Watch 机制实现配置热更新，无需 reload Nginx。请求处理时，APISIX 在 Rewrite 和 Access 阶段执行插件链，支持动态加载和卸载插件。控制面（Admin API + Dashboard）管理配置，数据面（APISIX 实例）处理流量，两者通过 etcd 解耦。APISIX 支持通过 Plugin Config 将一组插件打包复用，简化配置管理。

## 四、优缺点

**优点：**
- etcd 实时 Watch，配置变更毫秒级生效
- 支持多协议（HTTP/HTTPS/gRPC/TCP/UDP），协议覆盖全面
- 国内社区活跃，Apache 顶级项目，文档丰富
- Dashboard 提供可视化管理，降低使用门槛

**缺点：**
- 依赖 etcd 集群，增加运维复杂度
- 相比 Kong 历史较短，部分场景插件成熟度不及 Kong
- Lua 插件开发同样有学习成本

## 五、最佳实践

1. etcd 集群至少 3 节点，保障配置存储高可用
2. 使用 Plugin Config 模板化管理插件组合
3. 开启 Prometheus 插件 + Grafana 面板做监控
4. 利用 APISIX Ingress Controller 替代 K8s 原生 Ingress

## 六、常见陷阱

1. **etcd 集群故障导致配置无法更新**，需监控 etcd 健康状态
2. **Admin API 未加鉴权**，任何人可修改路由配置
3. **Route 优先级依赖创建顺序**，需手动设置 priority 字段
4. **大量 Route 时 etcd 存储压力增大**，需定期清理无用路由
