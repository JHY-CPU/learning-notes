# REST HTTP 通信

## 一、Feign 客户端

```java
@FeignClient(name = "order-service", fallbackFactory = OrderFallbackFactory.class)
public interface OrderClient {
    @GetMapping("/orders/{id}")
    Order getOrder(@PathVariable Long id);

    @PostMapping("/orders")
    Order createOrder(@RequestBody OrderDTO dto);
}

@Component
public class OrderFallbackFactory implements FallbackFactory<OrderClient> {
    @Override
    public OrderClient create(Throwable cause) {
        return new OrderClient() {
            @Override
            public Order getOrder(Long id) {
                return Order.defaultOrder(id);
            }
        };
    }
}
```

## 二、工作原理

Feign 是声明式 HTTP 客户端，通过接口注解定义远程调用，框架自动生成实现类。调用时 Feign 将方法参数序列化为 HTTP 请求，通过负载均衡器选择目标实例发送请求，收到响应后反序列化为返回对象。FallbackFactory 比 Fallback 更灵活，可以获取异常原因做差异化降级。Feign 集成 Hystrix/Sentinel 实现熔断，集成 Ribbon/LoadBalancer 实现负载均衡。

## 三、优缺点

**优点：**
- 声明式编程，代码简洁
- 与 Spring Cloud 生态无缝集成
- 支持熔断降级、负载均衡

**缺点：**
- HTTP 协议开销大于二进制协议（如 gRPC）
- JSON 序列化性能低于 Protobuf
- 接口变更需要同步更新客户端定义

## 四、最佳实践

1. 所有 Feign 调用必须配置超时（连接 3 秒，读取 5-10 秒）
2. 必须配置 FallbackFactory 处理降级
3. 接口版本通过 URL Path 管理（如 /v1/orders）
4. 统一错误码和响应格式（如 {code, message, data}）

## 五、完整 Feign 配置

```yaml
# 超时、重试、编码器完整配置
feign:
  client:
    config:
      default:
        connect-timeout: 3000
        read-timeout: 5000
        logger-level: BASIC
      order-service:          # 特定服务的配置
        connect-timeout: 3000
        read-timeout: 15000   # 订单服务可能较慢
  circuitbreaker:
    enabled: true
  compression:
    request:
      enabled: true
      mime-types: text/xml,application/xml,application/json
      min-request-size: 2048
    response:
      enabled: true

# 日志配置
logging:
  level:
    com.example.client.OrderClient: DEBUG
```

## 六、REST API 设计规范

```java
// 统一响应格式
@Data
public class Result<T> {
    private int code;          // 业务状态码
    private String message;    // 提示信息
    private T data;            // 业务数据
    private long timestamp;    // 时间戳

    public static <T> Result<T> success(T data) {
        return new Result<>(200, "success", data, System.currentTimeMillis());
    }

    public static <T> Result<T> fail(int code, String message) {
        return new Result<>(code, message, null, System.currentTimeMillis());
    }
}

// 分页查询
@GetMapping("/orders")
public Result<Page<Order>> listOrders(
        @RequestParam(defaultValue = "1") int page,
        @RequestParam(defaultValue = "20") int size,
        @RequestParam(required = false) String status) {
    Page<Order> orders = orderService.list(page, size, status);
    return Result.success(orders);
}

// RESTful URL 设计
// GET    /api/v1/orders          - 查询订单列表
// GET    /api/v1/orders/{id}     - 查询单个订单
// POST   /api/v1/orders          - 创建订单
// PUT    /api/v1/orders/{id}     - 更新订单
// DELETE /api/v1/orders/{id}     - 删除订单
// POST   /api/v1/orders/{id}/pay - 订单支付（动作）
```

## 七、常见陷阱

1. **无超时配置** - 下游服务慢导致调用方阻塞
2. **无降级逻辑** - 下游故障直接抛异常到前端
3. **Feign 接口参数不加 @PathVariable/@RequestBody 注解** - 序列化失败
4. **FallbackFactory 中的日志未记录异常原因** - 排障困难
5. **API 版本管理缺失** - 不兼容变更导致客户端崩溃
6. **响应格式不统一** - 前端解析困难
