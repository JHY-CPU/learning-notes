# Sentinel 详解

## 一、Sentinel 核心功能

```
Sentinel 功能:
├── 流量控制 - QPS/线程数限流
├── 熔断降级 - 慢调用/异常比例熔断
├── 热点参数 - 参数级限流
├── 系统保护 - 系统负载保护
└── 黑白名单 - 来源控制
```

## 二、Spring Boot 集成

```java
@SentinelResource(
    value = "getUser",
    blockHandler = "getUserBlockHandler",
    fallback = "getUserFallback"
)
public User getUser(Long id) {
    return userService.getUser(id);
}

// 限流降级
public User getUserBlockHandler(Long id, BlockException ex) {
    return User.defaultUser(id);
}

// 异常降级
public User getUserFallback(Long id, Throwable t) {
    return User.defaultUser(id);
}
```

## 三、流控规则

```java
// 代码配置规则
FlowRule rule = new FlowRule();
rule.setResource("getUser");
rule.setGrade(RuleConstant.FLOW_GRADE_QPS);
rule.setCount(100);  // QPS 限制 100
FlowRuleManager.loadRules(List.of(rule));
```

## 四、Sentinel 熔断规则

```java
// 慢调用熔断
DegradeRule slowRule = new DegradeRule("getUser")
    .setGrade(CircuitBreakerStrategy.SLOW_REQUEST_RATIO.getType())
    .setCount(0.5)                    // 慢调用比例阈值 50%
    .setSlowRatioThreshold(1000)      // 慢调用 RT 阈值 1000ms
    .setMinRequestAmount(10)          // 最小请求数
    .setStatIntervalMs(1000)          // 统计窗口 1 秒
    .setTimeWindow(5);                // 熔断时长 5 秒

// 异常比例熔断
DegradeRule errorRule = new DegradeRule("getUser")
    .setGrade(CircuitBreakerStrategy.ERROR_RATIO.getType())
    .setCount(0.5)                    // 异常比例阈值 50%
    .setMinRequestAmount(10)
    .setStatIntervalMs(1000)
    .setTimeWindow(5);

// 异常数熔断
DegradeRule countRule = new DegradeRule("getUser")
    .setGrade(CircuitBreakerStrategy.ERROR_COUNT.getType())
    .setCount(10)                     // 异常数阈值 10 次
    .setMinRequestAmount(1)
    .setStatIntervalMs(60000)         // 统计窗口 1 分钟
    .setTimeWindow(10);               // 熔断时长 10 秒

DegradeRuleManager.loadRules(List.of(slowRule, errorRule, countRule));
```

## 五、Sentinel 热点参数限流

```java
// 热点参数限流 - 针对特定参数值限流
@SentinelResource(value = "getProduct", blockHandler = "getProductBlockHandler")
public Product getProduct(Long categoryId, Long productId) {
    return productService.getProduct(productId);
}

// 热点规则 - 热门商品限流更严格
ParamFlowRule hotRule = new ParamFlowRule("getProduct")
    .setParamIdx(1)                   // 第 2 个参数 (productId)
    .setCount(50);                    // 默认 QPS 50

// 特定参数值的特殊限流
ParamFlowItem item = new ParamFlowItem()
    .setObject(1001L)                 // 热门商品 ID
    .setClassType(Long.class.getName())
    .setCount(10);                    // 该商品限流 10 QPS
hotRule.setParamFlowItemList(List.of(item));

ParamFlowRuleManager.loadRules(List.of(hotRule));
```

## 六、Sentinel + Nacos 持久化规则

```yaml
# Sentinel 规则持久化到 Nacos
spring:
  cloud:
    sentinel:
      datasource:
        flow:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-flow-rules
            groupId: SENTINEL_GROUP
            rule-type: flow
        degrade:
          nacos:
            server-addr: localhost:8848
            dataId: ${spring.application.name}-degrade-rules
            groupId: SENTINEL_GROUP
            rule-type: degrade
```

## 七、注意事项

1. **Sentinel Dashboard 提供可视化管理** - 实时监控 + 动态规则配置
2. **规则支持动态推送** - 通过 Nacos/Apollo 持久化和推送规则
3. **熔断策略要根据业务调整** - 支付接口用异常数，查询接口用慢调用比例
4. **降级逻辑要简洁高效** - 降级本身不能成为性能瓶颈
5. **监控规则触发情况** - Sentinel Dashboard 实时显示规则触发次数
6. **热点参数限流** - 防止个别热门数据（如秒杀商品）击穿系统
