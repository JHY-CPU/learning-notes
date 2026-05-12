# WebSocket 实时通信

## 一、Spring WebSocket

```java
@ServerEndpoint("/ws/{userId}")
@Component
public class WebSocketServer {
    private static Map<String, Session> sessions = new ConcurrentHashMap<>();

    @OnOpen
    public void onOpen(Session session, @PathParam("userId") String userId) {
        sessions.put(userId, session);
    }

    @OnMessage
    public void onMessage(String message, Session session) {
        // 处理消息
    }

    @OnClose
    public void onClose(Session session) {
        sessions.values().remove(session);
    }
}
```

## 二、工作原理

WebSocket 在 HTTP 握手后建立全双工 TCP 长连接，客户端和服务端可随时互相推送消息。服务端为每个客户端维护一个 Session 对象，存储在 ConcurrentHashMap 中。多实例部署时，服务端之间需要通过 Redis Pub/Sub 或 MQ 同步消息，确保连接在不同实例上的客户端也能互相通信。心跳机制通过 Ping/Pong 帧检测连接活性，长时间无数据时保持连接不被防火墙或代理断开。

## 三、优缺点

**优点：**
- 全双工通信，实时性极高
- 连接建立后无需重复握手，开销小
- 适合聊天、通知、实时数据推送

**缺点：**
- 连接管理复杂，需要处理重连和断线
- 多实例部署需要会话同步
- 长连接占用服务器资源

## 四、最佳实践**

1. 心跳间隔设 30-60 秒，检测连接活性
2. 使用 Redis Pub/Sub 做多实例消息广播
3. WebSocket 连接也需要 JWT 认证，在握手阶段验证 Token
4. 负载均衡使用 IP Hash 或 Cookie 粘性会话

## 五、多实例 WebSocket 同步

```java
// Redis Pub/Sub 实现多实例消息同步
@Component
public class WebSocketMessageBroker {
    @Autowired
    private StringRedisTemplate redisTemplate;

    // 本地 Session 存储
    private static final Map<String, Session> LOCAL_SESSIONS = new ConcurrentHashMap<>();

    // 发送消息到指定用户
    public void sendToUser(String userId, String message) {
        Session session = LOCAL_SESSIONS.get(userId);
        if (session != null && session.isOpen()) {
            // 本地有连接，直接发送
            sendMessage(session, message);
        } else {
            // 通过 Redis 广播到其他实例
            redisTemplate.convertAndSend("ws:user:" + userId, message);
        }
    }

    // 订阅 Redis 消息
    @Bean
    public RedisMessageListenerContainer container(RedisConnectionFactory factory) {
        RedisMessageListenerContainer container = new RedisMessageListenerContainer();
        container.setConnectionFactory(factory);
        container.addMessageListener((message, pattern) -> {
            String userId = extractUserId(new String(pattern));
            String payload = new String(message.getBody());
            Session session = LOCAL_SESSIONS.get(userId);
            if (session != null && session.isOpen()) {
                sendMessage(session, payload);
            }
        }, new PatternTopic("ws:user:*"));
        return container;
    }
}
```

## 六、WebSocket + STOMP 协议

```java
// STOMP 配置 - 更高级的 WebSocket 通信
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.enableSimpleBroker("/topic", "/queue");  // 消息代理
        config.setApplicationDestinationPrefixes("/app"); // 应用前缀
        config.setUserDestinationPrefix("/user");         // 用户前缀
    }

    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws").setAllowedOriginPatterns("*")
            .withSockJS();  // SockJS 降级支持
    }
}

// 消息控制器
@Controller
public class ChatController {
    @MessageMapping("/chat.sendMessage")
    @SendTo("/topic/public")
    public ChatMessage sendMessage(@Payload ChatMessage message) {
        return message;
    }

    @MessageMapping("/chat.private")
    @SendToUser("/queue/messages")
    public ChatMessage privateMessage(@Payload ChatMessage message) {
        return message;
    }
}
```

## 七、常见陷阱

1. **无心跳机制** - 连接被 Nginx 等代理超时断开（proxy_read_timeout 默认 60s）
2. **多实例部署无消息同步** - 部分客户端收不到消息
3. **连接数无限制** - 被恶意创建大量连接耗尽资源
4. **WebSocket 无认证** - 任何人可连接并接收消息，握手阶段必须验证 Token
5. **Nginx WebSocket 代理配置缺失** - 需要配置 Upgrade 和 Connection 头
6. **内存泄漏** - Session 关闭时未清理 Map 中的引用
