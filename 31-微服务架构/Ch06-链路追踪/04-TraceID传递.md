# TraceID 传递

## 一、HTTP Header 传递

```java
// Feign 传递 TraceID
@FeignClient(name = "order-service")
public interface OrderClient {
    @GetMapping("/orders/{id}")
    Order getOrder(@PathVariable Long id,
                   @RequestHeader("X-Trace-Id") String traceId);
}

// Filter 恢复 TraceID
@Component
public class TraceFilter implements Filter {
    @Override
    public void doFilter(ServletRequest request, ServletResponse response, FilterChain chain) {
        HttpServletRequest req = (HttpServletRequest) request;
        String traceId = req.getHeader("X-Trace-Id");
        if (traceId != null) {
            TraceContext.setTraceId(traceId);
        }
        try {
            chain.doFilter(request, response);
        } finally {
            TraceContext.clear();
        }
    }
}
```

## 二、MQ 传递

```java
// MQ 消息携带 TraceID
ProducerRecord<String, String> record = new ProducerRecord<>("topic", message);
record.headers().add("traceId", TraceContext.getTraceId().getBytes());
```

## 三、注意事项

1. **所有跨服务调用都要传递 TraceID**
2. **MQ 消息也要携带 TraceID**
3. **MDC 存储 TraceID 方便日志输出**
4. **异步场景要特别注意上下文传递**
5. **TraceID 使用 UUID 或 Snowflake**
