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

## 四、注意事项

1. **Kong 性能很高**，适合大规模 API
2. **插件生态丰富**
3. **Lua 开发有一定门槛**
4. **配置可以用声明式 YAML**
5. **适合独立部署的 API 网关**
