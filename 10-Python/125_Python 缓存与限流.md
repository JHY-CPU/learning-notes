# Python 缓存与限流


## ⚡ Python 缓存与限流


cachetools (LRU/TTL)、functools.lru_cache、Redis 缓存策略、限流算法实现 (令牌桶/滑动窗口)、aiocache 异步缓存。


## functools.lru_cache


```
// ========== functools.lru_cache ==========
from functools import lru_cache
import time

# lru_cache: 最近最少使用缓存
# 适合: 纯函数/计算密集型/递归优化

@lru_cache(maxsize=128)
def fibonacci(n):
    """计算斐波那契数 (带缓存)"""
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

# 首次调用: 计算并缓存
start = time.time()
print(fibonacci(35))  # 9227465
print(f"耗时: {time.time() - start:.4f}s")

# 再次调用: 直接从缓存返回
start = time.time()
print(fibonacci(35))
print(f"缓存命中: {time.time() - start:.4f}s")  # 几乎 0 秒

# 缓存信息:
print(fibonacci.cache_info())
# CacheInfo(hits=32, misses=36, maxsize=128, currsize=36)

# 清除缓存:
fibonacci.cache_clear()
print(fibonacci.cache_info())

# maxsize=None: 无限缓存
@lru_cache(maxsize=None)
def expensive_function(x):
    return x ** x
```


## cachetools 库


```
// ========== cachetools ==========
# pip install cachetools

from cachetools import Cache, LRUCache, TTLCache, cached
import time

# 1. LRUCache: 最近最少使用
cache = LRUCache(maxsize=100)

@cached(cache)
def get_data(key):
    """数据会被缓存 (LRU 策略)"""
    print(f"计算: {key}")
    return key * 2

print(get_data(5))  # 计算 10
print(get_data(5))  # 缓存命中,直接返回

# 2. TTLCache: 带过期时间的缓存
ttl_cache = TTLCache(maxsize=100, ttl=5)  # 5 秒后过期

@cached(ttl_cache)
def get_weather(city):
    print(f"请求天气: {city}")
    return {"city": city, "temp": 25}  # 模拟 API 调用

print(get_weather("Beijing"))  # 请求
print(get_weather("Beijing"))  # 缓存 (5 秒内)
time.sleep(6)
print(get_weather("Beijing"))  # 过期,重新请求

# 3. 其他缓存策略:
from cachetools import LFUCache, FIFOCache, RRCache

# LFUCache: 最不经常使用
# FIFOCache: 先进先出
# RRCache: 随机替换

# 4. 自定义 key
def make_key(args, kwargs):
    return f"{args}:{sorted(kwargs.items())}"

@cached(LRUCache(maxsize=50), key=make_key)
def query_db(sql, params=None):
    # SQL 查询缓存
    pass
```


## Redis 缓存


```
// ========== Redis 缓存 ==========
# pip install redis

import redis
import json
from functools import wraps

# 连接 Redis
cache = redis.Redis(
    host="localhost",
    port=6379,
    db=0,
    decode_responses=True,  # 自动解码为字符串
)

# 基本操作:
cache.set("key", "value")
cache.setex("key_ttl", 3600, "1 小时后过期")  # 带过期时间
value = cache.get("key")

# ========== 缓存装饰器 ==========
def redis_cache(ttl=300):
    """Redis 缓存装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存 key
            key = f"{func.__name__}:{args}:{sorted(kwargs.items())}"

            # 尝试获取缓存
            cached = cache.get(key)
            if cached:
                print("缓存命中")
                return json.loads(cached)

            # 执行函数
            result = func(*args, **kwargs)

            # 存入缓存
            cache.setex(key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@redis_cache(ttl=60)
def get_user(user_id):
    """模拟数据库查询"""
    print(f"查询数据库: {user_id}")
    return {"id": user_id, "name": f"User-{user_id}"}

# 第一次: 查询数据库
user = get_user(1)
# 第二次: 缓存命中
user = get_user(1)

# ========== 批量操作 ==========
pipe = cache.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.expire("key1", 3600)
pipe.execute()  # 一次网络请求执行所有命令
```


## 限流算法


```
// ========== 令牌桶 ==========
import time
from collections import deque

class TokenBucket:
    """令牌桶限流"""

    def __init__(self, rate, capacity):
        self.rate = rate          # 每秒生成令牌数
        self.capacity = capacity   # 桶容量
        self.tokens = capacity     # 当前令牌数
        self.last_refill = time.time()

    def _refill(self):
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(
            self.capacity,
            self.tokens + elapsed * self.rate
        )
        self.last_refill = now

    def consume(self, count=1):
        """消费令牌,返回是否允许通过"""
        self._refill()
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

bucket = TokenBucket(rate=10, capacity=20)

# 使用:
for i in range(30):
    if bucket.consume():
        print(f"请求 {i}: 通过")
    else:
        print(f"请求 {i}: 限流")

# ========== 滑动窗口 ==========
from collections import defaultdict

class SlidingWindow:
    """滑动窗口限流"""

    def __init__(self, window_size=60, max_requests=100):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = deque()

    def allow(self):
        now = time.time()
        # 移除窗口外的请求
        while self.requests and self.requests[0] < now - self.window_size:
            self.requests.popleft()

        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False
```


## aiocache 异步缓存


```
// ========== aiocache ==========
# pip install aiocache

import asyncio
from aiocache import Cache, cached
from aiocache.serializers import JsonSerializer

# aiocache: 异步缓存库,支持 Redis/Memory/Memcached

# 内存缓存:
cache = Cache(Cache.MEMORY)

# Redis 缓存:
redis_cache = Cache(
    Cache.REDIS,
    endpoint="localhost",
    port=6379,
    serializer=JsonSerializer(),
)

# ========== 装饰器方式 ==========
@cached(
    ttl=300,
    cache=Cache.REDIS,
    key_builder=lambda f, args, kwargs: f"my_cache:{args[0]}",
    serializer=JsonSerializer(),
)
async def get_user_data(user_id):
    """模拟异步数据库查询"""
    await asyncio.sleep(1)
    return {"id": user_id, "name": f"User-{user_id}"}

async def main():
    # 第一次: 等待 1 秒
    data = await get_user_data(1)
    print(data)

    # 第二次: 立即返回
    data = await get_user_data(1)
    print(data)

# ========== 缓存模式 ==========
# Cache Aside:
async def get_product(product_id):
    # 先查缓存
    cached = await cache.get(f"product:{product_id}")
    if cached:
        return cached

    # 缓存未命中,查数据库
    product = await db.fetch_product(product_id)
    await cache.set(f"product:{product_id}", product, ttl=300)
    return product

# 更新: 删除缓存
async def update_product(product_id, data):
    await db.update_product(product_id, data)
    await cache.delete(f"product:{product_id}")
```


> **Note:** 💡 缓存要点: lru_cache 函数级缓存; cachetools LRU/TTL; Redis 分布式缓存; 读多写少 + 过期时间; 令牌桶/滑动窗口限流; aiocache 异步缓存。


## 练习


<!-- Converted from: 125_Python 缓存与限流.html -->
