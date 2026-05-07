# Resilience4j 详解

## 一、核心功能

```java
// 熔断器
@CircuitBreaker(name = "userService", fallbackMethod = "fallback")
public User getUser(Long id) {
    return userService.getUser(id);
}

// 重试
@Retry(name = "userService", fallbackMethod = "fallback")
public User getUser(Long id) {
    return userService.getUser(id);
}

// 限流
@RateLimiter(name = "userService")
public User getUser(Long id) {
    return userService.getUser(id);
}

// 舱壁隔离
@Bulkhead(name = "userService")
public User getUser(Long id) {
    return userService.getUser(id);
}

// 降级方法
public User fallback(Long id, Throwable t) {
    return User.defaultUser(id);
}
```

## 二、注意事项

1. **Resilience4j 轻量级，无外部依赖**
2. **函数式编程风格**
3. **与 Spring Boot 集成良好**
4. **支持多种保护机制组合**
5. **配置要根据实际调优**
