# Redlock算法

## 一、概念说明

Redlock是Redis作者提出的分布式锁算法，通过在多个独立的Redis节点上获取锁来避免单点故障。

## 二、算法流程

```
1. 获取当前时间戳
2. 依次向N个节点请求锁
3. 获取锁的条件：在多数节点(N/2+1)获取成功
4. 锁有效时间 = 过期时间 - 获取锁耗时
5. 如果获取失败，向所有节点释放锁
```

## 三、Python实现

```python
import redis
import time
import uuid

class Redlock:
    def __init__(self, nodes):
        self.nodes = [redis.Redis(**node) for node in nodes]
        self.quorum = len(nodes) // 2 + 1
    
    def acquire(self, key, ttl=30000):
        uid = str(uuid.uuid4())
        start = time.time() * 1000
        
        acquired = 0
        for node in self.nodes:
            try:
                if node.set(key, uid, nx=True, px=ttl):
                    acquired += 1
            except:
                pass
        
        elapsed = time.time() * 1000 - start
        validity = ttl - elapsed - 2  # 时钟漂移补偿
        
        if acquired >= self.quorum and validity > 0:
            return uid, validity
        else:
            self.release(key, uid)
            return None, 0
    
    def release(self, key, uid):
        for node in self.nodes:
            try:
                lua = "if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('del',KEYS[1]) else return 0 end"
                node.eval(lua, 1, key, uid)
            except:
                pass

# 使用
nodes = [
    {'host': '192.168.1.1', 'port': 6379},
    {'host': '192.168.1.2', 'port': 6379},
    {'host': '192.168.1.3', 'port': 6379},
]
redlock = Redlock(nodes)
uid, validity = redlock.acquire("lock:resource")
```

## 四、注意事项

1. **节点独立**：Redis节点必须独立，无主从关系
2. **时钟同步**：节点间时间偏差不能太大
3. **至少3节点**：保证多数派
4. **争议**：Redlock在分布式领域有争议
5. **简单场景**：单节点分布式锁足够大多数场景
