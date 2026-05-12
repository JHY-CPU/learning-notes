# API 网关最佳实践

## 一、实践清单

```yaml
功能:
  □ 统一认证授权
  □ 限流熔断
  □ 日志追踪
  □ 灰度发布
  □ 协议转换

架构:
  □ 网关至少 2 节点
  □ 负载均衡前置
  □ 配置中心管理路由
  □ 监控告警完善
```

## 二、选型建议

| 场景 | 推荐 |
|------|------|
| Spring 技术栈 | Spring Cloud Gateway |
| 高性能 API | Kong / APISIX |
| Service Mesh | Envoy |
| 轻量级 | Nginx |

## 三、工作原理

API 网关是微服务架构的流量入口，所有外部请求先经过网关再转发到内部服务。网关在请求生命周期中依次执行：协议转换（如 WebSocket 转 HTTP）、认证鉴权、限流熔断、请求路由、负载均衡、日志追踪、响应转换。网关将这些横切关注点从业务服务中剥离，使下游服务专注于业务逻辑。网关通常部署在负载均衡器之后，至少 2 个节点做高可用。

## 四、优缺点

**优点：**
- 统一入口简化客户端调用，减少客户端复杂度
- 横切关注点集中处理，避免各服务重复实现
- 支持灰度发布、协议转换等高级特性

**缺点：**
- 网关成为单点，需重点保障高可用
- 网关引入额外延迟（通常 1-5ms）
- 网关配置管理复杂度随路由数量增长

## 五、网关安全最佳实践

```java
// API Key + JWT 双重认证
@Component
public class AuthFilter implements GlobalFilter, Ordered {
    private static final Set<String> PUBLIC_PATHS = Set.of(
        "/api/public/", "/actuator/health", "/api/auth/login"
    );

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String path = exchange.getRequest().getPath().value();

        // 公开接口直接放行
        if (PUBLIC_PATHS.stream().anyMatch(path::startsWith)) {
            return chain.filter(exchange);
        }

        // 校验 JWT
        String authHeader = exchange.getRequest().getHeaders().getFirst("Authorization");
        if (authHeader == null || !authHeader.startsWith("Bearer ")) {
            return unauthorized(exchange, "Missing or invalid Authorization header");
        }

        try {
            Claims claims = parseToken(authHeader.substring(7));
            // 注入用户信息到请求头
            ServerHttpRequest request = exchange.getRequest().mutate()
                .header("X-User-Id", claims.getSubject())
                .header("X-User-Role", claims.get("role", String.class))
                .build();
            return chain.filter(exchange.mutate().request(request).build());
        } catch (Exception e) {
            return unauthorized(exchange, "Token expired or invalid");
        }
    }
}
```

## 六、网关性能优化

```yaml
# 连接池和线程优化
spring:
  cloud:
    gateway:
      httpclient:
        pool:
          type: elastic
          max-connections: 1000
          max-idle-time: 30s
          acquire-timeout: 5000
        connect-timeout: 1000
        response-timeout: 10s
      compression:
        enabled: true
        mime-types: application/json,application/xml
        min-size: 1024
```

```java
// 缓存路由信息，减少注册中心查询
spring:
  cloud:
    gateway:
      discovery:
        locator:
          enabled: true
          lower-case-service-id: true
          cache-duration: 30s    # 缓存 30 秒
```

## 七、网关高可用架构

```
                    用户
                     │
              ┌──────▼──────┐
              │   DNS/CDN   │
              └──────┬──────┘
              ┌──────▼──────┐
              │  负载均衡器   │ (SLB/Nginx)
              └──────┬──────┘
         ┌───────────┼───────────┐
    ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
    │Gateway-1│ │Gateway-2│ │Gateway-3│
    │ (2C4G)  │ │ (2C4G)  │ │ (2C4G)  │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
    ┌────▼───────────▼───────────▼────┐
    │          注册中心集群             │
    └─────────────────────────────────┘

关键配置:
- 至少 3 个网关实例，跨可用区部署
- HPA 自动伸缩（CPU > 70% 扩容）
- 健康检查 + 自动故障转移
- 灰度发布支持
```

## 八、常见陷阱

1. **在网关中写业务逻辑** - 导致网关耦合过重，难以维护，业务应下沉到服务
2. **网关节点资源不足** - 成为性能瓶颈，建议 2C4G 起步
3. **路由配置散落在各处** - 没有统一管理和版本控制，应集中到配置中心
4. **未监控网关指标** - 故障时无法快速定位，必须有 QPS/延迟/错误率监控
5. **限流阈值设置不合理** - 正常用户被误杀，需要根据历史数据调优
6. **忽略网关自身的容灾** - 网关挂了全站不可用，必须做高可用
