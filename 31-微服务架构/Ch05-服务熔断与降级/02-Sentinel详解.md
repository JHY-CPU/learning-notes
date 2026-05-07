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

## 四、注意事项

1. **Sentinel Dashboard 提供可视化管理**
2. **规则支持动态推送**
3. **熔断策略要根据业务调整**
4. **降级逻辑要简洁高效**
5. **监控规则触发情况**
