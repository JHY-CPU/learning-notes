# 网关 WebSocket 支持

## 一、核心概念

WebSocket 是一种全双工通信协议，适用于实时推送、在线聊天、股票行情等场景。在微服务架构中，API 网关需要支持 WebSocket 协议的转发，同时保证认证、限流、负载均衡等功能对 WebSocket 连接同样生效。

### 1.1 WebSocket 与 HTTP 的区别

```
HTTP（半双工）:
  客户端 ──请求──→ 服务器
  客户端 ←──响应── 服务器
  （每次通信都需要建立连接）

WebSocket（全双工）:
  客户端 ══握手═══→ 服务器   (HTTP Upgrade)
  客户端 ←═══握手═══ 服务器
  客户端 ═══双向通信═══ 服务器  (长连接)
```

### 1.2 网关 WebSocket 架构

```
  客户端                    API 网关                  WebSocket 服务
    │                         │                          │
    │── HTTP Upgrade ────────►│                          │
    │                         │── Upgrade (转发) ───────►│
    │                         │◄── 101 Switching ────────│
    │◄── 101 Switching ───────│                          │
    │                         │                          │
    │═══ 双向消息通信 ═════════════════════════════════════│
    │                         │                          │
    │── Close ───────────────►│── Close ────────────────►│
```

## 二、Spring Cloud Gateway WebSocket 配置

### 2.1 基本路由配置

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: websocket-route
          uri: ws://websocket-service
          predicates:
            - Path=/ws/**
          filters:
            - StripPrefix=1
```

### 2.2 基于服务发现的 WebSocket 路由

```yaml
spring:
  cloud:
    gateway:
      routes:
        - id: ws-chat-service
          uri: lb:ws://chat-service
          predicates:
            - Path=/ws/chat/**
        - id: ws-notification-service
          uri: lb:ws://notification-service
          predicates:
            - Path=/ws/notifications/**
```

### 2.3 WebSocket 连接认证

```java
@Component
@Order(-2)
public class WebSocketAuthFilter implements GlobalFilter {

    @Override
    public Mono<Void> filter(ServerWebExchange exchange,
                              GatewayFilterChain chain) {
        ServerHttpRequest request = exchange.getRequest();

        // WebSocket 握手时进行认证
        if (isWebSocketUpgrade(request)) {
            String token = request.getQueryParams()
                .getFirst("token");
            if (token == null || !validateToken(token)) {
                exchange.getResponse()
                    .setStatusCode(HttpStatus.UNAUTHORIZED);
                return exchange.getResponse().setComplete();
            }

            // 将用户信息传递给后端
            String userId = extractUserId(token);
            exchange.getRequest().mutate()
                .header("X-User-Id", userId);
        }

        return chain.filter(exchange);
    }

    private boolean isWebSocketUpgrade(
            ServerHttpRequest request) {
        return "websocket".equals(
            request.getHeaders().getUpgrade());
    }
}
```

## 三、WebSocket 负载均衡

### 3.1 会话粘性

WebSocket 连接是长连接，负载均衡需要考虑会话粘性：

```nginx
# Nginx 配置会话粘性
upstream websocket_servers {
    ip_hash;  # 基于 IP 哈希，同一客户端连接同一后端
    server ws-service-1:8080;
    server ws-service-2:8080;
    server ws-service-3:8080;
}

server {
    location /ws/ {
        proxy_pass http://websocket_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }
}
```

### 3.2 自定义负载均衡策略

```java
@Component
public class WebSocketLoadBalancer
        implements ReactorServiceInstanceLoadBalancer {

    @Override
    public Mono<Response<ServiceInstance>> choose(
            Request request) {
        ServerHttpRequest httpRequest =
            ((RequestAdapter) request).getHttpRequest();

        // 从请求中获取连接标识
        String connectionId = httpRequest.getHeaders()
            .getFirst("X-Connection-Id");

        if (connectionId != null) {
            // 基于连接 ID 哈希路由到固定实例
            int hash = Math.abs(connectionId.hashCode());
            return selectByHash(hash);
        }

        // 降级为轮询
        return roundRobin();
    }
}
```

## 四、WebSocket 心跳管理

### 4.1 Ping/Pong 机制

```java
@Configuration
public class WebSocketHeartbeatConfig {

    @Bean
    public WebSocketClient webSocketClient() {
        return new StandardWebSocketClient();
    }

    @Bean
    public WebSocketConnectionManager connectionManager() {
        WebSocketConnectionManager manager =
            new WebSocketConnectionManager(
                webSocketClient(),
                new MyWebSocketHandler(),
                "ws://backend-service/ws"
            );
        manager.setAutoStartup(true);
        return manager;
    }
}

@Component
public class ConnectionHealthMonitor {

    private final Map<String, WebSocketSession> sessions =
        new ConcurrentHashMap<>();

    /**
     * 定期检查连接健康状态
     */
    @Scheduled(fixedRate = 30000)
    public void checkConnections() {
        sessions.entrySet().removeIf(entry -> {
            WebSocketSession session = entry.getValue();
            if (!session.isOpen()) {
                log.info("Connection closed: {}",
                    entry.getKey());
                return true;
            }
            return false;
        });
        log.info("Active WebSocket connections: {}",
            sessions.size());
    }
}
```

## 五、消息广播与推送

### 5.1 基于 Redis 的消息广播

```java
@Component
public class WebSocketBroadcastService {

    @Autowired
    private StringRedisTemplate redisTemplate;

    @Autowired
    private RedisMessageListenerContainer listenerContainer;

    /**
     * 订阅 Redis 频道，广播消息到所有 WebSocket 连接
     */
    @PostConstruct
    public void subscribe() {
        listenerContainer.addMessageListener(
            (message, pattern) -> {
                String channel =
                    new String(message.getChannel());
                String body = new String(message.getBody());
                broadcastToLocalClients(channel, body);
            },
            new ChannelTopic("ws:broadcast")
        );
    }

    /**
     * 发送消息到所有网关实例
     */
    public void publish(String topic, String message) {
        redisTemplate.convertAndSend("ws:broadcast",
            JsonUtil.toJson(Map.of("topic", topic,
                "message", message)));
    }
}
```

## 六、最佳实践

1. **WebSocket 路由使用 `ws://` 或 `wss://` 协议前缀**
2. **网关要支持 WebSocket Upgrade 请求的转发**
3. **负载均衡要考虑会话粘性**，避免连接频繁切换
4. **心跳检测连接是否存活**，及时清理无效连接
5. **WebSocket 也要做认证**，在握手阶段完成鉴权
6. **设置合理的超时时间**，避免连接长时间空闲占用资源
7. **使用 Redis Pub/Sub 实现跨实例消息广播**
8. **监控 WebSocket 连接数**，防止连接泄漏
