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

## 二、工作原理

Resilience4j 基于函数式编程和装饰器模式实现。熔断器维护一个滑动窗口，统计最近 N 次调用的失败率，超过阈值后进入 OPEN 状态拒绝请求，经过等待时间后进入 HALF_OPEN 状态放行少量请求探测，成功则关闭熔断。重试机制在异常发生后按配置的间隔和退避策略重试。限流器使用令牌桶算法控制 QPS。舱壁隔离通过信号量或独立线程池限制并发数，防止一个依赖的故障耗尽所有资源。各组件可任意组合，按顺序装饰目标方法。

## 三、优缺点

**优点：**
- 轻量级纯 Java 库，无外部依赖（不需要 Zuul/Eureka 等）
- 函数式编程风格，API 简洁优雅
- 支持 CircuitBreaker、Retry、RateLimiter、Bulkhead、TimeLimiter 五大组件
- 可与 Spring Boot 通过 Starter 无缝集成

**缺点：**
- 没有可视化控制台，规则调整需修改配置
- 不支持集群级流控，仅限单机
- 与 Sentinel 相比功能覆盖面稍窄

## 四、最佳实践

1. 熔断器阈值根据实际延迟和错误率数据调优，不要使用默认值
2. 重试只用于幂等操作，非幂等操作禁止重试
3. 舱壁隔离为核心服务分配独立线程池，非核心服务用信号量
4. 组合使用时注意顺序：TimeLimiter -> CircuitBreaker -> Retry

## 五、常见陷阱

1. **熔断窗口太小导致频繁切换状态**，产生请求抖动
2. **重试加重故障**，下游不可用时大量重试加剧雪崩
3. **线程池隔离资源浪费**，每个依赖分配独立线程池但利用率低
4. **fallback 方法签名不匹配**，导致降级不生效
