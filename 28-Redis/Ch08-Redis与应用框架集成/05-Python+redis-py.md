# Python + redis-py

## 一、概念说明

redis-py是Python官方推荐的Redis客户端，支持同步和异步操作。

## 二、基本使用

```python
import redis

# 连接
r = redis.Redis(
    host='192.168.1.100',
    port=6379,
    password='yourpassword',
    db=0,
    decode_responses=True
)

# 基本操作
r.set('key', 'value', ex=3600)
value = r.get('key')
r.delete('key')

# Hash操作
r.hset('user:1', 'name', '张三')
r.hget('user:1', 'name')
r.hgetall('user:1')

# List操作
r.lpush('queue', 'task1')
r.rpop('queue')
r.lrange('queue', 0, -1)

# Set操作
r.sadd('tags', 'python', 'redis')
r.smembers('tags')

# ZSet操作
r.zadd('rank', {'player1': 100, 'player2': 200})
r.zrevrange('rank', 0, -1, withscores=True)
```

## 三、连接池

```python
import redis

pool = redis.ConnectionPool(
    host='192.168.1.100',
    port=6379,
    max_connections=20,
    decode_responses=True
)

r = redis.Redis(connection_pool=pool)
```

## 四、Pipeline

```python
pipe = r.pipeline()
for i in range(1000):
    pipe.set(f'key:{i}', f'value:{i}')
pipe.execute()
```

## 五、异步支持

```python
import redis.asyncio as aioredis

async def main():
    r = aioredis.Redis(host='192.168.1.100', port=6379)
    await r.set('key', 'value')
    value = await r.get('key')
    await r.close()
```

## 六、注意事项

1. **decode_responses**：自动解码为字符串
2. **连接池**：多线程使用连接池
3. **异常处理**：捕获ConnectionError
## 七、Pub/Sub使用

```python
import redis
import threading
import json

r = redis.Redis()

def subscribe_handler():
    """订阅处理"""
    pubsub = r.pubsub()
    pubsub.subscribe('notifications')
    
    for message in pubsub.listen():
        if message['type'] == 'message':
            data = json.loads(message['data'])
            print(f"收到通知: {data}")

# 启动订阅线程
thread = threading.Thread(target=subscribe_handler, daemon=True)
thread.start()

# 发布消息
r.publish('notifications', json.dumps({
    'type': 'info',
    'message': '系统维护通知'
}))
```

## 八、Stream消费者组

```python
import redis
import json

r = redis.Redis()

# 创建消费者组
try:
    r.xgroup_create('orders', 'order_group', id='0', mkstream=True)
except redis.ResponseError:
    pass  # 组已存在

def consume_orders():
    """消费订单消息"""
    while True:
        messages = r.xreadgroup(
            'order_group', 'consumer1',
            {'orders': '>'},
            count=10, block=5000
        )
        
        for stream, msgs in messages:
            for msg_id, data in msgs:
                try:
                    order = json.loads(data[b'data'])
                    print(f"处理订单: {order}")
                    
                    # 确认消息
                    r.xack('orders', 'order_group', msg_id)
                except Exception as e:
                    print(f"处理失败: {e}")

# 启动消费
consume_orders()
```

## 九、Redis锁实现

```python
import redis
import time
from uuid import uuid4

r = redis.Redis()

class RedisLock:
    def __init__(self, key, expire=30):
        self.key = f"lock:{key}"
        self.expire = expire
        self.token = str(uuid4())
    
    def acquire(self, timeout=10):
        end_time = time.time() + timeout
        while time.time() < end_time:
            if r.set(self.key, self.token, nx=True, ex=self.expire):
                return True
            time.sleep(0.1)
        return False
    
    def release(self):
        lua = """
        if redis.call('get', KEYS[1]) == ARGV[1] then
            return redis.call('del', KEYS[1])
        end
        return 0
        """
        return r.eval(lua, 1, self.key, self.token)

# 使用
lock = RedisLock("resource:1001")
if lock.acquire():
    try:
        print("获取锁成功")
    finally:
        lock.release()
```
