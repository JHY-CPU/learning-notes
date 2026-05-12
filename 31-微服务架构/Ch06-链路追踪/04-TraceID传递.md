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

## 三、线程池传递 TraceID

```java
// 解决 TraceID 在线程池中丢失的问题
@Component
public class TracedThreadPoolExecutor {
    @Bean("tracedExecutor")
    public Executor tracedExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(10);
        executor.setMaxPoolSize(20);
        executor.setQueueCapacity(100);
        // 装饰器模式 - 捕获 MDC 上下文
        executor.setTaskDecorator(runnable -> {
            Map<String, String> contextMap = MDC.getCopyOfContextMap();
            return () -> {
                if (contextMap != null) MDC.setContextMap(contextMap);
                try {
                    runnable.run();
                } finally {
                    MDC.clear();
                }
            };
        });
        executor.initialize();
        return executor;
    }
}

// 使用
@Autowired @Qualifier("tracedExecutor")
private Executor executor;

public CompletableFuture<User> getUserAsync(Long id) {
    return CompletableFuture.supplyAsync(() -> userService.getUser(id), executor);
}
```

## 四、Feign 拦截器传递

```java
// 自动传递 TraceID 到 Feign 调用
@Component
public class FeignTraceInterceptor implements RequestInterceptor {
    @Override
    public void apply(RequestTemplate template) {
        // 从 MDC 获取 TraceID
        String traceId = MDC.get("traceId");
        if (traceId != null) {
            template.header("X-Trace-Id", traceId);
        }

        // SkyWalking 自动注入 sw8 Header
        // Zipkin/Sleuth 自动注入 X-B3-* Headers
    }
}
```

## 五、MQ 消息 TraceID 传递

```java
// Kafka Producer - 携带 TraceID
@Component
public class TracedKafkaProducer {
    @Autowired
    private KafkaTemplate<String, String> kafkaTemplate;

    public void send(String topic, String message) {
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, message);
        // 添加 TraceID 到 Header
        String traceId = MDC.get("traceId");
        if (traceId != null) {
            record.headers().add("X-Trace-Id", traceId.getBytes(StandardCharsets.UTF_8));
        }
        kafkaTemplate.send(record);
    }
}

// Kafka Consumer - 恢复 TraceID
@Component
public class TracedKafkaConsumer {
    @KafkaListener(topics = "order-events")
    public void onMessage(ConsumerRecord<String, String> record) {
        // 从 Header 恢复 TraceID
        Header traceIdHeader = record.headers().lastHeader("X-Trace-Id");
        if (traceIdHeader != null) {
            String traceId = new String(traceIdHeader.value(), StandardCharsets.UTF_8);
            MDC.put("traceId", traceId);
        }
        try {
            processMessage(record.value());
        } finally {
            MDC.clear();
        }
    }
}
```

## 六、注意事项

1. **所有跨服务调用都要传递 TraceID** - HTTP/MQ/RPC 无一例外
2. **MQ 消息也要携带 TraceID** - 在消息 Header 中携带
3. **MDC 存储 TraceID 方便日志输出** - Logback 配置 %X{traceId}
4. **异步场景要特别注意上下文传递** - 线程池需要装饰器传递 MDC
5. **TraceID 使用 UUID 或 Snowflake** - 保证全局唯一
6. **OpenTelemetry 的 Context API** - 提供标准的上下文传递机制
