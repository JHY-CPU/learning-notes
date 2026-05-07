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

## 二、注意事项

1. **Feign 声明式调用最简单**
2. **必须配置超时和熔断**
3. **接口版本管理**
4. **请求响应格式统一**
5. **错误码规范化**
