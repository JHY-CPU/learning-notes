# 分布式Session

## 一、概念说明

分布式Session将用户会话信息存储在Redis中，实现多服务器间的Session共享，解决Session不一致问题。

## 二、实现方案

### Spring Session

```yaml
# application.yml
spring:
  session:
    store-type: redis
    redis:
      flush-mode: on_save
      namespace: spring:session
    timeout: 1800s
```

### Node.js + express-session

```javascript
const session = require('express-session');
const RedisStore = require('connect-redis').default;

app.use(session({
    store: new RedisStore({
        client: redisClient,
        prefix: 'sess:'
    }),
    secret: 'your-secret-key',
    resave: false,
    saveUninitialized: false,
    cookie: { maxAge: 1800000 }
}));
```

## 三、Redis存储格式

```bash
# Session数据
HSET spring:session:sessions:{id} attr:user '{"name":"张三"}'
HSET spring:session:sessions:{id} creationTime 1704067200
HSET spring:session:sessions:{id} maxInactiveInterval 1800

# 过期索引
ZADD spring:session:sessions:expires 1704069000 {sessionId}

# Session清理
# Redis自动清理过期的Session
```

## 四、注意事项

1. **序列化**：选择合适的序列化方式（JSON/二进制）
2. **过期时间**：合理设置Session超时时间
3. **安全性**：使用HTTPS、HttpOnly、Secure Cookie
4. **性能**：Session操作有网络开销
5. **故障处理**：Redis不可用时的降级策略
