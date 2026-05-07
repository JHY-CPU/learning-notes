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

## 五、注意事项

1. **Spring Cloud Gateway 是响应式**，不要用阻塞代码
2. **lb:// 表示负载均衡**
3. **Predicate 可以组合使用**
4. **Filter 有顺序**，用 @Order 控制
5. **生产环境要配置超时和重试**
