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

## 五、Python Session管理

```python
import redis
import json
import time
from uuid import uuid4

r = redis.Redis()

class RedisSessionManager:
    def __init__(self, prefix="session", expire=1800):
        self.prefix = prefix
        self.expire = expire
    
    def create_session(self, user_id, data=None):
        """创建Session"""
        session_id = str(uuid4())
        session_key = f"{self.prefix}:{session_id}"
        
        session_data = {
            'user_id': user_id,
            'created': time.time(),
            'last_accessed': time.time(),
            'data': data or {}
        }
        
        r.setex(session_key, self.expire, json.dumps(session_data))
        return session_id
    
    def get_session(self, session_id):
        """获取Session"""
        session_key = f"{self.prefix}:{session_id}"
        data = r.get(session_key)
        
        if not data:
            return None
        
        session = json.loads(data)
        
        # 更新访问时间
        session['last_accessed'] = time.time()
        r.setex(session_key, self.expire, json.dumps(session))
        
        return session
    
    def update_session(self, session_id, data):
        """更新Session"""
        session_key = f"{self.prefix}:{session_id}"
        existing = self.get_session(session_id)
        
        if existing:
            existing['data'].update(data)
            r.setex(session_key, self.expire, json.dumps(existing))
            return True
        return False
    
    def delete_session(self, session_id):
        """删除Session"""
        session_key = f"{self.prefix}:{session_id}"
        r.delete(session_key)
    
    def get_user_sessions(self, user_id):
        """获取用户的所有Session"""
        sessions = []
        pattern = f"{self.prefix}:*"
        
        for key in r.scan_iter(match=pattern, count=100):
            data = r.get(key)
            if data:
                session = json.loads(data)
                if session.get('user_id') == user_id:
                    sessions.append({
                        'session_id': key.decode().split(':')[-1],
                        'created': session['created'],
                        'last_accessed': session['last_accessed']
                    })
        
        return sessions
    
    def cleanup_expired(self):
        """清理过期Session（可选）"""
        # Redis自动处理Key过期，此方法用于主动清理
        pattern = f"{self.prefix}:*"
        deleted = 0
        
        for key in r.scan_iter(match=pattern, count=100):
            ttl = r.ttl(key)
            if ttl < 0:  # 已过期
                r.delete(key)
                deleted += 1
        
        return deleted

# 使用示例
manager = RedisSessionManager()

# 创建Session
session_id = manager.create_session(user_id=1001, data={'role': 'admin'})
print(f"Session ID: {session_id}")

# 获取Session
session = manager.get_session(session_id)
print(f"用户ID: {session['user_id']}")

# 更新Session
manager.update_session(session_id, {'last_action': 'view_page'})

# 删除Session（登出）
manager.delete_session(session_id)
```

## 六、Session安全增强

```python
class SecureSessionManager(RedisSessionManager):
    """安全增强的Session管理"""
    
    def create_session(self, user_id, ip_address=None, user_agent=None, data=None):
        """创建Session并绑定安全信息"""
        session_id = str(uuid4())
        session_key = f"{self.prefix}:{session_id}"
        
        session_data = {
            'user_id': user_id,
            'created': time.time(),
            'last_accessed': time.time(),
            'ip_address': ip_address,
            'user_agent': user_agent,
            'data': data or {}
        }
        
        r.setex(session_key, self.expire, json.dumps(session_data))
        
        # 记录用户Session索引
        r.sadd(f"user:sessions:{user_id}", session_id)
        r.expire(f"user:sessions:{user_id}", self.expire)
        
        return session_id
    
    def validate_session(self, session_id, ip_address=None, user_agent=None):
        """验证Session安全性"""
        session = self.get_session(session_id)
        
        if not session:
            return False
        
        # 检查IP变化（可选，根据安全要求）
        if ip_address and session.get('ip_address') != ip_address:
            print(f"警告: Session {session_id} IP地址变化")
            # 可以选择删除Session或要求重新验证
        
        return True
    
    def limit_concurrent_sessions(self, user_id, max_sessions=5):
        """限制并发Session数"""
        session_ids = r.smembers(f"user:sessions:{user_id}")
        
        if len(session_ids) >= max_sessions:
            # 删除最旧的Session
            oldest = None
            oldest_time = float('inf')
            
            for sid in session_ids:
                sid = sid.decode()
                session = self.get_session(sid)
                if session and session['created'] < oldest_time:
                    oldest_time = session['created']
                    oldest = sid
            
            if oldest:
                self.delete_session(oldest)
                r.srem(f"user:sessions:{user_id}", oldest)
```
