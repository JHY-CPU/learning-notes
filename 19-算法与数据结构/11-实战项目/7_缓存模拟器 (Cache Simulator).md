# 缓存模拟器 (Cache Simulator)

## 项目需求与功能分析

缓存是计算机系统性能优化的核心手段。本项目模拟 LRU、LFU、FIFO 三种经典缓存淘汰策略，帮助深入理解缓存的工作原理和适用场景。

### 核心功能

- LRU（最近最少使用）缓存策略实现
- LFU（最不经常使用）缓存策略实现
- FIFO（先进先出）缓存策略实现
- 缓存命中率统计与可视化
- 访问序列模拟与对比分析
- 缓存状态实时展示

### 应用场景

- CPU 缓存设计
- 数据库查询缓存
- Web 浏览器缓存
- CDN 内容分发
- 操作系统页面置换

## 核心算法原理

### LRU (Least Recently Used)

淘汰最久未被访问的数据。使用哈希表 + 双向链表实现 O(1) 的访问和淘汰。

- 访问时：将节点移到链表头部
- 淘汰时：删除链表尾部节点

### LFU (Least Frequently Used)

淘汰访问频率最低的数据。使用哈希表 + 频率桶实现。

- 维护每个 key 的访问频率
- 淘汰频率最低的 key（频率相同时淘汰最旧的）

### FIFO (First In, First Out)

最先进入缓存的数据最先被淘汰。使用队列实现。

## 完整代码实现

```python
from collections import OrderedDict, defaultdict, deque
from typing import Any, Optional, Dict, List
from dataclasses import dataclass


class LRUCache:
    """LRU 缓存 - 使用 OrderedDict"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: OrderedDict = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key) -> Optional[Any]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
        else:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)  # 淘汰最旧的
            self.cache[key] = value

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def status(self):
        return {
            'type': 'LRU',
            'capacity': self.capacity,
            'size': len(self.cache),
            'keys': list(self.cache.keys()),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.hit_rate():.2%}"
        }


class LFUNode:
    def __init__(self, key, value, freq=1):
        self.key = key
        self.value = value
        self.freq = freq


class LFUCache:
    """LFU 缓存"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[Any, LFUNode] = {}
        self.freq_map: Dict[int, deque] = defaultdict(deque)
        self.min_freq = 0
        self.hits = 0
        self.misses = 0

    def get(self, key) -> Optional[Any]:
        if key not in self.cache:
            self.misses += 1
            return None
        self.hits += 1
        node = self.cache[key]
        self._update_freq(node)
        return node.value

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.cache:
            node = self.cache[key]
            node.value = value
            self._update_freq(node)
            return
        if len(self.cache) >= self.capacity:
            self._evict()
        node = LFUNode(key, value)
        self.cache[key] = node
        self.freq_map[1].append(key)
        self.min_freq = 1

    def _update_freq(self, node):
        old_freq = node.freq
        self.freq_map[old_freq].remove(node.key)
        if not self.freq_map[old_freq]:
            del self.freq_map[old_freq]
            if self.min_freq == old_freq:
                self.min_freq += 1
        node.freq += 1
        self.freq_map[node.freq].append(node.key)

    def _evict(self):
        key = self.freq_map[self.min_freq].popleft()
        if not self.freq_map[self.min_freq]:
            del self.freq_map[self.min_freq]
        del self.cache[key]

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def status(self):
        return {
            'type': 'LFU',
            'capacity': self.capacity,
            'size': len(self.cache),
            'keys': list(self.cache.keys()),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.hit_rate():.2%}"
        }


class FIFOCache:
    """FIFO 缓存"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[Any, Any] = {}
        self.queue: deque = deque()
        self.hits = 0
        self.misses = 0

    def get(self, key) -> Optional[Any]:
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            return
        if len(self.cache) >= self.capacity:
            oldest = self.queue.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.queue.append(key)

    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def status(self):
        return {
            'type': 'FIFO',
            'capacity': self.capacity,
            'size': len(self.cache),
            'keys': list(self.cache.keys()),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.hit_rate():.2%}"
        }


def simulate_access_pattern(pattern: str, size: int = 100) -> List[int]:
    """生成不同类型的访问序列"""
    import random
    if pattern == 'uniform':
        return [random.randint(0, 20) for _ in range(size)]
    elif pattern == 'zipf':
        # Zipf 分布：少数元素被频繁访问
        items = list(range(30))
        weights = [1.0 / (i + 1) for i in range(30)]
        return random.choices(items, weights=weights, k=size)
    elif pattern == 'temporal':
        # 时间局部性：最近访问的元素更可能被再次访问
        result = []
        recent = deque(maxlen=5)
        for _ in range(size):
            if recent and random.random() < 0.7:
                result.append(random.choice(list(recent)))
            else:
                item = random.randint(0, 20)
                result.append(item)
                recent.append(item)
        return result
    else:
        return [random.randint(0, 20) for _ in range(size)]


def compare_caches(capacity: int = 5, pattern: str = 'zipf', size: int = 200):
    """对比三种缓存策略"""
    access_seq = simulate_access_pattern(pattern, size)

    lru = LRUCache(capacity)
    lfu = LFUCache(capacity)
    fifo = FIFOCache(capacity)

    for key in access_seq:
        val = f"val_{key}"
        for cache in [lru, lfu, fifo]:
            result = cache.get(key)
            if result is None:
                cache.put(key, val)

    print(f"\n访问模式: {pattern} | 容量: {capacity} | 访问次数: {size}")
    print(f"{'策略':<8} {'命中':<8} {'缺失':<8} {'命中率':<10}")
    print("-" * 36)
    for cache in [lru, lfu, fifo]:
        s = cache.status()
        print(f"{s['type']:<8} {s['hits']:<8} {s['misses']:<8} {s['hit_rate']:<10}")
```

## 测试用例

```python
import unittest

class TestCaches(unittest.TestCase):
    def test_lru_basic(self):
        c = LRUCache(2)
        c.put(1, 'a'); c.put(2, 'b')
        self.assertEqual(c.get(1), 'a')
        c.put(3, 'c')  # 淘汰 key 2
        self.assertIsNone(c.get(2))
        self.assertEqual(c.get(3), 'c')

    def test_lfu_basic(self):
        c = LFUCache(2)
        c.put(1, 'a'); c.put(2, 'b')
        c.get(1); c.get(1)  # key 1 频率更高
        c.put(3, 'c')  # 淘汰 key 2
        self.assertIsNone(c.get(2))
        self.assertEqual(c.get(1), 'a')

    def test_fifo_basic(self):
        c = FIFOCache(2)
        c.put(1, 'a'); c.put(2, 'b')
        c.get(1)  # FIFO 不管访问，只管插入顺序
        c.put(3, 'c')  # 淘汰 key 1
        self.assertIsNone(c.get(1))

    def test_hit_rate(self):
        c = LRUCache(3)
        for i in range(10): c.put(i, i)
        for i in range(7, 10): c.get(i)
        self.assertGreater(c.hit_rate(), 0)

    def test_access_patterns(self):
        for p in ['uniform', 'zipf', 'temporal']:
            seq = simulate_access_pattern(p, 50)
            self.assertEqual(len(seq), 50)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **ARC 缓存**：自适应替换缓存，结合 LRU 和 LFU
2. **Clock 算法**：LRU 的近似实现，开销更低
3. **多级缓存**：L1 / L2 / L3 分级缓存模拟
4. **缓存穿透 / 雪崩**：模拟异常场景
5. **分布式缓存**：一致性哈希分片
6. **TTL 过期**：支持缓存条目的自动过期
7. **可视化仪表盘**：实时展示缓存状态和命中率曲线
