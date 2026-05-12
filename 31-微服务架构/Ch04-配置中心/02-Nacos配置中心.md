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

## 三、工作原理

Nacos 配置中心使用 Data ID + Group + Namespace 三级定位一个配置。服务启动时根据这些维度向 Nacos 请求配置内容，Nacos 从内置数据库（默认 Derby，生产用 MySQL）读取配置返回。运行期间，客户端通过长轮询（默认 30 秒）监听配置变更，Nacos 收到轮询请求后如果有变更立即返回，否则挂起直到超时。@RefreshScope 注解标记的 Bean 在配置变更时会被销毁重建，@NacosValue 注解的字段通过 Listener 回调原地更新。

## 四、优缺点

**优点：**
- 注册中心 + 配置中心一体化，减少组件数量
- 国内社区活跃，文档和案例丰富
- 支持多种配置格式（YAML、Properties、JSON）

**缺点：**
- 长轮询模式有一定延迟（通常秒级）
- 大量配置变更时推送风暴可能影响性能
- 权限管理相对简单，不适合严格审计场景

## 五、最佳实践

1. 命名空间按环境划分（dev/test/prod），绝对不要跨环境共享
2. 共享配置使用 shared-configs 引用，避免重复定义
3. 敏感配置配合 Jasypt 或 Nacos 自带加密功能
4. 配置变更后通过 @RefreshScope 或事件监听实现动态生效

## 六、常见陷阱

1. **@RefreshScope 导致 Bean 重建**，有状态的 Bean 会丢失状态
2. **长轮询超时配置不当**，导致变更感知延迟
3. **Shared configs 优先级不清晰**，可能覆盖业务配置
4. **Namespace 未配置导致读到错误环境的配置**
