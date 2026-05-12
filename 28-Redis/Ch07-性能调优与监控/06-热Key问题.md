# 热Key问题

## 一、概念说明

热Key是指访问频率极高的Key，集中在少数节点上，导致该节点负载过高。

## 二、发现热Key

```bash
# 方法1：monitor统计
redis-cli MONITOR | head -n 10000 | awk '{print $4}' | sort | uniq -c | sort -rn

# 方法2：redis-cli --hotkeys（需LFU策略）
redis-cli --hotkeys

# 方法3：客户端统计
# 在应用层统计Key访问频率

# 方法4：proxy统计
# 通过代理层统计
```

## 三、解决方案

```bash
# 1. 本地缓存
# 热Key缓存到应用本地
@Cacheable(value="hot", key="#id")

# 2. Key分片
# product:1001 → product:1001:1, product:1001:2, product:1001:3
shard = random.randint(1, 3)
key = f"product:1001:{shard}"

# 3. 读写分离
# 读操作分散到从节点

# 4. 二级缓存
# 本地缓存 + Redis缓存
```

## 四、注意事项

1. **及时发现**：监控热Key的产生
2. **动态调整**：根据访问模式动态分片
3. **本地缓存一致性**：TTL要短
## 五、热Key详细检测

```python
import redis
import time
from collections import defaultdict

r = redis.Redis()

def detect_hot_keys_via_monitor(duration=10):
    """通过MONITOR命令检测热Key"""
    print(f"监控 {duration} 秒...")
    
    pubsub = r.monitor()
    key_counts = defaultdict(int)
    
    start = time.time()
    for command in pubsub.listen():
        if time.time() - start > duration:
            break
        
        if 'command' in command:
            parts = command['command'].split()
            if len(parts) >= 2:
                cmd = parts[0].upper()
                key = parts[1]
                # 只统计读命令
                if cmd in ('GET', 'HGET', 'HGETALL', 'LRANGE', 'SMEMBERS', 'ZRANGE'):
                    key_counts[key] += 1
    
    # 排序并输出
    sorted_keys = sorted(key_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n热Key排名 (Top 20):")
    for key, count in sorted_keys[:20]:
        print(f"  {key}: {count} 次/秒")
    
    return sorted_keys

# 执行检测
hot_keys = detect_hot_keys_via_monitor(duration=10)
```

## 六、本地缓存完整实现

```python
import redis
import time
import threading
from functools import lru_cache

r = redis.Redis()

class LocalCacheWithRedis:
    """本地缓存 + Redis 二级缓存"""
    
    def __init__(self, local_ttl=10, max_size=1000):
        self.local_ttl = local_ttl
        self.max_size = max_size
        self.local_cache = {}
        self.local_cache_time = {}
        self.lock = threading.Lock()
    
    def get(self, key):
        """获取数据（本地缓存优先）"""
        # 检查本地缓存
        with self.lock:
            if key in self.local_cache:
                if time.time() - self.local_cache_time.get(key, 0) < self.local_ttl:
                    return self.local_cache[key]
                else:
                    # 本地缓存过期
                    del self.local_cache[key]
                    del self.local_cache_time[key]
        
        # 从Redis获取
        value = r.get(key)
        if value:
            decoded = value.decode()
            
            # 写入本地缓存
            with self.lock:
                if len(self.local_cache) >= self.max_size:
                    # LRU淘汰
                    oldest_key = min(self.local_cache_time, key=self.local_cache_time.get)
                    del self.local_cache[oldest_key]
                    del self.local_cache_time[oldest_key]
                
                self.local_cache[key] = decoded
                self.local_cache_time[key] = time.time()
            
            return decoded
        
        return None
    
    def set(self, key, value, redis_ttl=3600):
        """设置数据"""
        # 写入Redis
        r.setex(key, redis_ttl, value)
        
        # 更新本地缓存
        with self.lock:
            self.local_cache[key] = value
            self.local_cache_time[key] = time.time()
    
    def delete(self, key):
        """删除数据"""
        r.delete(key)
        
        with self.lock:
            if key in self.local_cache:
                del self.local_cache[key]
                del self.local_cache_time[key]
    
    def invalidate_local(self, key):
        """只失效本地缓存"""
        with self.lock:
            if key in self.local_cache:
                del self.local_cache[key]
                del self.local_cache_time[key]

# 使用
cache = LocalCacheWithRedis(local_ttl=10)

# 第一次从Redis获取
value = cache.get("product:hot")

# 后续从本地缓存获取
value = cache.get("product:hot")  # 本地命中
```

## 七、热Key分片策略

```python
class HotKeySharding:
    """热Key分片"""
    
    def __init__(self, shard_count=4):
        self.shard_count = shard_count
        self.r = redis.Redis()
    
    def _get_shard_key(self, key, shard_id):
        return f"{key}:shard:{shard_id}"
    
    def get(self, key):
        """随机读取一个分片"""
        import random
        shard_id = random.randint(0, self.shard_count - 1)
        shard_key = self._get_shard_key(key, shard_id)
        return self.r.get(shard_key)
    
    def set(self, key, value, ttl=3600):
        """写入所有分片"""
        pipe = self.r.pipeline()
        for i in range(self.shard_count):
            shard_key = self._get_shard_key(key, i)
            pipe.setex(shard_key, ttl, value)
        pipe.execute()
    
    def delete(self, key):
        """删除所有分片"""
        pipe = self.r.pipeline()
        for i in range(self.shard_count):
            shard_key = self._get_shard_key(key, i)
            pipe.delete(shard_key)
        pipe.execute()

# 使用
sharding = HotKeySharding(shard_count=4)
sharding.set("product:hot", '{"id":1,"name":"热门商品"}')
value = sharding.get("product:hot")
```
