# WebSocket 支持

## 一、Spring Cloud Gateway WebSocket

```yaml
spring:
  cloud:
    gateway:
      routes:
      - id: websocket-route
        uri: ws://websocket-service
        predicates:
        - Path=/ws/**
```

## 二、注意事项

1. **WebSocket 需要长连接**
2. **网关要支持 WebSocket 升级**
3. **负载均衡要考虑会话粘性**
4. **心跳检测连接是否存活**
5. **WebSocket 也要做认证**
