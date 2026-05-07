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

## 二、注意事项

1. **WebSocket 适合实时推送**
2. **连接管理要做好**
3. **心跳检测防止连接断开**
4. **负载均衡要考虑会话粘性**
5. **WebSocket 也要做认证**
