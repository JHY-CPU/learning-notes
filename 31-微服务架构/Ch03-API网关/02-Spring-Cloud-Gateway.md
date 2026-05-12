# Spring Cloud Gateway

## 一、核心概念

```
Route (路由) = Predicate (断言) + Filter (过滤器)

Predicate: 匹配请求条件
Filter: 处理请求/响应
```

## 二、基本配置

```yaml
spring:
  cloud:
    gateway:
      routes:
      - id: user-service
        uri: lb://user-service
        predicates:
        - Path=/api/users/**
        filters:
        - StripPrefix=1
        - name: CircuitBreaker
          args:
            name: user-cb
            fallbackUri: forward:/fallback

      - id: order-service
        uri: lb://order-service
        predicates:
        - Path=/api/orders/**
```

## 三、常用 Filter

```yaml
filters:
- StripPrefix=1           # 去掉前缀
- AddRequestHeader=X-Request-Id, 123  # 添加请求头
- RequestRateLimiter=...  # 限流
- CircuitBreaker=...      # 熔断
- RewritePath=/api/(?<segment>.*), /${segment}  # 路径重写
```

## 四、全局过滤器

```java
@Component
public class AuthFilter implements GlobalFilter {
    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        String token = exchange.getRequest().getHeaders().getFirst("Authorization");
        if (!validToken(token)) {
            exchange.getResponse().setStatusCode(HttpStatus.UNAUTHORIZED);
            return exchange.getResponse().setComplete();
        }
        return chain.filter(exchange);
    }
}
```

## 五、Predicate 断言详解

```yaml
spring:
  cloud:
    gateway:
      routes:
      # Path 匹配
      - id: path-route
        uri: lb://user-service
        predicates:
        - Path=/api/users/**

      # Header 匹配
      - id: header-route
        uri: lb://user-service-v2
        predicates:
        - Header=X-Version, v2

      # 多条件组合
      - id: composite-route
        uri: lb://order-service
        predicates:
        - Path=/api/orders/**
        - Method=POST,PUT
        - Header=Authorization,Bearer*
        - After=2024-01-01T00:00:00+08:00

      # Cookie 匹配（灰度发布）
      - id: cookie-route
        uri: lb://user-service-canary
        predicates:
        - Cookie=track,canary
```

## 六、自定义 Filter 示例

```java
// 请求日志过滤器
@Component
public class RequestLoggingFilter implements GlobalFilter, Ordered {
    private static final Logger log = LoggerFactory.getLogger(RequestLoggingFilter.class);

    @Override
    public Mono<Void> filter(ServerWebExchange exchange, GatewayFilterChain chain) {
        long startTime = System.currentTimeMillis();
        ServerHttpRequest request = exchange.getRequest();

        // 注入 TraceID
        String traceId = Optional.ofNullable(request.getHeaders().getFirst("X-Trace-Id"))
            .orElse(UUID.randomUUID().toString());

        exchange.getRequest().mutate().header("X-Trace-Id", traceId);

        return chain.filter(exchange).then(Mono.fromRunnable(() -> {
            long duration = System.currentTimeMillis() - startTime;
            ServerHttpResponse response = exchange.getResponse();
            log.info("[{}] {} {} -> {} ({}ms)",
                traceId,
                request.getMethod(),
                request.getURI(),
                response.getStatusCode(),
                duration);
        }));
    }

    @Override
    public int getOrder() { return -1; }
}

// 请求限流 Key 解析器
@Component
public class ApiKeyResolver implements KeyResolver {
    @Override
    public Mono<String> resolve(ServerWebExchange exchange) {
        // 按 API 路径 + 用户 ID 限流
        String userId = exchange.getRequest().getHeaders().getFirst("X-User-Id");
        String path = exchange.getRequest().getPath().value();
        return Mono.just(userId != null ? userId + ":" + path : path);
    }
}
```

## 七、Gateway 超时与重试配置

```yaml
spring:
  cloud:
    gateway:
      default-filters:
      - name: Retry
        args:
          retries: 3
          statuses: BAD_GATEWAY, SERVICE_UNAVAILABLE
          methods: GET
          backoff:
            firstBackoff: 100ms
            maxBackoff: 500ms
            factor: 2

      httpclient:
        connect-timeout: 1000
        response-timeout: 5s
        pool:
          type: elastic
          max-connections: 500
          acquire-timeout: 4000
```

## 八、注意事项

1. **Spring Cloud Gateway 是响应式** - 使用 Reactor Netty，不要用阻塞代码（如 JDBC）
2. **lb:// 表示负载均衡** - 通过注册中心发现服务实例
3. **Predicate 可以组合使用** - 支持 Path + Method + Header 多条件匹配
4. **Filter 有顺序** - 用 @Order 或实现 Ordered 接口控制执行顺序
5. **生产环境要配置超时和重试** - connect-timeout + response-timeout + retry
6. **响应式编程不要阻塞** - 数据库操作用 R2DBC，不要用 JPA/JDBC
