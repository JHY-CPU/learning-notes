# 系统设计算法 (System Design Algorithms)

## 一、一致性哈希 (Consistent Hashing)

### 1.1 原理

传统哈希取模在节点增减时会导致大量数据迁移。一致性哈希将节点和数据映射到一个哈希环上，数据存储在顺时针方向最近的节点。

**虚拟节点：** 每个物理节点映射多个虚拟节点到环上，解决数据倾斜。

```python
import hashlib
import bisect

class ConsistentHash:
    def __init__(self, replicas=100):
        self.replicas = replicas
        self.ring = []
        self.nodes = {}

    def _hash(self, key):
        return int(hashlib.md5(key.encode()).hexdigest(), 16)

    def add_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}:{i}")
            bisect.insort(self.ring, h)
            self.nodes[h] = node

    def remove_node(self, node):
        for i in range(self.replicas):
            h = self._hash(f"{node}:{i}")
            self.ring.remove(h)
            del self.nodes[h]

    def get_node(self, key):
        if not self.ring: return None
        h = self._hash(key)
        idx = bisect.bisect_right(self.ring, h) % len(self.ring)
        return self.nodes[self.ring[idx]]
```

---

## 二、布隆过滤器 (Bloom Filter)

### 2.1 原理

用多个哈希函数将元素映射到一个位数组。查询时所有位都为1则**可能存在**，任一位为0则**一定不存在**。

**特点：** 只有假阳性，没有假阴性。

```python
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, size=10000, hash_count=3):
        self.size = size
        self.hash_count = hash_count
        self.bits = bitarray(size)
        self.bits.setall(0)

    def add(self, item):
        for i in range(self.hash_count):
            idx = mmh3.hash(item, i) % self.size
            self.bits[idx] = 1

    def contains(self, item):
        for i in range(self.hash_count):
            idx = mmh3.hash(item, i) % self.size
            if not self.bits[idx]:
                return False
        return True  # 可能存在（假阳性）
```

### 2.2 最优参数

- 位数组大小：$m = -\frac{n \ln p}{(\ln 2)^2}$
- 哈希函数个数：$k = \frac{m}{n} \ln 2$

其中 $n$ 为预期元素数，$p$ 为期望假阳性率。

---

## 三、限流器 (Rate Limiter)

### 3.1 令牌桶算法

```python
import time

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate          # 每秒添加的令牌数
        self.capacity = capacity  # 桶容量
        self.tokens = capacity
        self.last_time = time.time()

    def allow(self):
        now = time.time()
        elapsed = now - self.last_time
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_time = now
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False
```

### 3.2 滑动窗口计数器

```python
class SlidingWindowCounter:
    def __init__(self, window_size, max_requests):
        self.window_size = window_size
        self.max_requests = max_requests
        self.requests = []  # (timestamp, count)

    def allow(self):
        import time
        now = time.time()
        # 清理过期数据
        self.requests = [(t, c) for t, c in self.requests
                        if now - t < self.window_size]
        total = sum(c for _, c in self.requests)
        if total < self.max_requests:
            self.requests.append((now, 1))
            return True
        return False
```

---

## 四、BitMap

### 4.1 用位图进行大规模去重

```python
class BitMap:
    def __init__(self, size):
        self.size = size
        self.bits = [0] * ((size >> 5) + 1)  # 每个int存32位

    def set(self, pos):
        idx, offset = pos >> 5, pos & 31
        self.bits[idx] |= (1 << offset)

    def get(self, pos):
        idx, offset = pos >> 5, pos & 31
        return bool(self.bits[idx] & (1 << offset))

    def clear(self, pos):
        idx, offset = pos >> 5, pos & 31
        self.bits[idx] &= ~(1 << offset)
```

---

## 五、HyperLogLog (基数估计)

### 5.1 原理

估计集合中不同元素的数量，使用固定 $O(1)$ 空间。

核心思想：观察哈希值前导零的个数来估计基数。

```python
import hashlib
import math

class HyperLogLog:
    def __init__(self, p=14):
        self.p = p
        self.m = 1 << p
        self.registers = [0] * self.m
        self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, value):
        return int(hashlib.sha256(str(value).encode()).hexdigest(), 16)

    def add(self, value):
        h = self._hash(value)
        idx = h & (self.m - 1)
        w = h >> self.p
        self.registers[idx] = max(self.registers[idx],
                                   len(bin(w)) - 2)  # 前导零个数+1

    def count(self):
        Z = sum(2 ** -r for r in self.registers)
        E = self.alpha * self.m * self.m / Z
        return int(E)
```

---

## 六、分层时间轮 (Hierarchical Timing Wheel)

用于高效管理大量定时任务。

```python
class TimingWheel:
    def __init__(self, slot_num=60):
        self.slot_num = slot_num
        self.current = 0
        self.slots = [[] for _ in range(slot_num)]

    def add_task(self, delay, task):
        slot = (self.current + delay) % self.slot_num
        self.slots[slot].append(task)

    def tick(self):
        tasks = self.slots[self.current]
        self.slots[self.current] = []
        self.current = (self.current + 1) % self.slot_num
        return tasks
```

---

## 七、面试中的系统设计算法

| 算法 | 应用场景 | 核心特点 |
|------|---------|---------|
| 一致性哈希 | 分布式缓存 | 节点变化最小迁移 |
| 布隆过滤器 | 缓存穿透、垃圾邮件 | 只有假阳性 |
| 限流器 | API限流 | 令牌桶/漏桶 |
| BitMap | 大规模去重 | 1bit/元素 |
| HyperLogLog | UV统计 | 固定内存 |
| 时间轮 | 定时任务 | $O(1)$ 插入触发 |
| LSM Tree | 数据库写优化 | 批量写入 |
| Skip List | 有序数据结构 | $O(\log n)$ 操作 |

---

## 八、复杂度分析

| 数据结构 | 空间 | 插入 | 查询 |
|---------|------|------|------|
| 一致性哈希 | $O(n \cdot r)$ | $O(\log nr)$ | $O(\log nr)$ |
| 布隆过滤器 | $O(m)$ bits | $O(k)$ | $O(k)$ |
| BitMap | $O(n/8)$ | $O(1)$ | $O(1)$ |
| HyperLogLog | $O(2^p)$ | $O(1)$ | $O(2^p)$ |
