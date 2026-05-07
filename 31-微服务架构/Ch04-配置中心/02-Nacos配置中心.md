# Nacos 配置中心

## 一、基本使用

```yaml
spring:
  cloud:
    nacos:
      config:
        server-addr: localhost:8848
        namespace: dev
        group: DEFAULT_GROUP
        file-extension: yaml
        shared-configs:
          - data-id: common.yaml
            group: DEFAULT_GROUP
            refresh: true
```

## 二、动态刷新

```java
@RestController
@RefreshScope  // 支持配置动态刷新
public class ConfigController {

    @Value("${app.feature.enabled:false}")
    private boolean featureEnabled;

    @GetMapping("/config")
    public Map<String, Object> getConfig() {
        return Map.of("featureEnabled", featureEnabled);
    }
}
```

## 三、注意事项

1. **@RefreshScope 实现动态刷新**
2. **命名空间隔离不同环境**
3. **共享配置减少重复**
4. **配置变更会推送事件**
5. **敏感配置使用加密**
