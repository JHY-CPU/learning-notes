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
4. **超时设置**：设置socket_timeout
