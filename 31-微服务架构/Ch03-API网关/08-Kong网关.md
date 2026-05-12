# Kong 网关

## 一、Kong 简介

Kong 是基于 OpenResty (Nginx + Lua) 的高性能 API 网关。

```bash
# Docker 启动 Kong
docker run -d --name kong \
  -e "KONG_DATABASE=off" \
  -e "KONG_DECLARATIVE_CONFIG=/etc/kong/kong.yml" \
  -p 8000:8000 \
  -p 8443:8443 \
  kong:latest
```

## 二、核心概念

```
Service  ->  Route  ->  Plugin
  │          │           │
 服务       路由        插件
```

## 三、常用插件

| 插件 | 功能 |
|------|------|
| rate-limiting | 限流 |
| jwt | JWT 认证 |
| cors | 跨域 |
| prometheus | 监控 |
| request-transformer | 请求转换 |

## 四、工作原理

Kong 基于 OpenResty（Nginx + LuaJIT）构建，请求到达后先经过 Nginx 的 access 阶段，Kong 在此阶段加载 Lua 插件链，按顺序执行认证、限流、日志等插件逻辑。Service 定义上游服务地址，Route 定义匹配规则（路径、方法、Header 等），Plugin 绑定到 Service 或 Route 上。DB-less 模式下所有配置通过声明式 YAML 文件加载，适合 GitOps 管理。Kong Manager 提供 Web 管理界面，Admin API 支持动态增删改查路由和插件。

## 五、优缺点

**优点：**
- 基于 Nginx，性能极高，单节点可处理数万 QPS
- 插件生态丰富，覆盖认证、限流、日志、转换等常见需求
- 支持 DB-less 模式，配置即代码

**缺点：**
- Lua 开发门槛较高，自定义插件需要学习 Lua
- 企业版功能（如 Dev Portal、RBAC）需要付费
- 配置管理依赖 PostgreSQL 或 DB-less 文件，各有运维成本

## 六、最佳实践

1. 生产环境至少 2 个 Kong 节点做高可用
2. 使用声明式配置 + Git 管理路由和插件
3. 优先使用社区插件，避免重复造轮子
4. Prometheus 插件接入监控系统，实时观测网关指标

## 七、常见陷阱

1. **插件执行顺序影响结果**，认证插件必须在限流插件之前
2. **PostgreSQL 数据库成为单点**，需要做主从或高可用
3. **Route 匹配优先级不明确**，导致请求匹配到错误的路由
4. **热更新配置时注意并发安全**，避免请求丢失
