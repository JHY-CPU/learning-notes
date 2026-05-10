# 设计数据结构专题 (Design Problems)

## 一、设计思路

### 1.1 设计题的核心方法

1. **明确需求：** 确认所有操作和时间复杂度要求
2. **选择底层结构：** 数组、链表、哈希表、堆、树
3. **权衡时间空间：** 用空间换时间或反之
4. **考虑并发：** 是否需要线程安全

---

## 二、经典设计题

### 2.1 LRU缓存 (LeetCode 146)

**思路：** 哈希表 + 双向链表。哈希表 $O(1)$ 查找，双向链表维护访问顺序。

```python
class LRUCache:
    class Node:
        def __init__(self, key=0, val=0):
            self.key, self.val = key, val
            self.prev = self.next = None

    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}
        self.head = self.Node()
        self.tail = self.Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node):
        node.next = self.head.next
        node.prev = self.head
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        if key not in self.cache: return -1
        node = self.cache[key]
        self._remove(node)
        self._add_front(node)
        return node.val

    def put(self, key, value):
        if key in self.cache:
            self._remove(self.cache[key])
        node = self.Node(key, value)
        self.cache[key] = node
        self._add_front(node)
        if len(self.cache) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.cache[lru.key]
```

### 2.2 LFU缓存 (LeetCode 460)

**思路：** 哈希表 + 双向链表 + 频率桶。

```python
from collections import defaultdict, OrderedDict

class LFUCache:
    def __init__(self, capacity):
        self.cap = capacity
        self.cache = {}  # key -> (value, freq)
        self.freq_map = defaultdict(OrderedDict)  # freq -> OrderedDict(key)
        self.min_freq = 0

    def _update_freq(self, key):
        val, freq = self.cache[key]
        del self.freq_map[freq][key]
        if not self.freq_map[freq]:
            del self.freq_map[freq]
            if self.min_freq == freq:
                self.min_freq += 1
        self.freq_map[freq+1][key] = None
        self.cache[key] = (val, freq+1)

    def get(self, key):
        if key not in self.cache: return -1
        self._update_freq(key)
        return self.cache[key][0]

    def put(self, key, value):
        if self.cap <= 0: return
        if key in self.cache:
            self.cache[key] = (value, self.cache[key][1])
            self._update_freq(key)
            return
        if len(self.cache) >= self.cap:
            evict_key, _ = self.freq_map[self.min_freq].popitem(last=False)
            del self.cache[evict_key]
        self.cache[key] = (value, 1)
        self.freq_map[1][key] = None
        self.min_freq = 1
```

### 2.3 实现Trie (LeetCode 208)

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for c in word:
            if c not in node.children:
                node.children[c] = Trie()
            node = node.children[c]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def _find(self, word):
        node = self
        for c in word:
            if c not in node.children: return None
            node = node.children[c]
        return node
```

### 2.4 最小栈 (LeetCode 155)

```python
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        self.min_stack.append(
            min(val, self.min_stack[-1]) if self.min_stack else val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def get_min(self):
        return self.min_stack[-1]
```

### 2.5 数据流的中位数 (LeetCode 295)

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.lo = []  # 大顶堆（最大元素在顶部）
        self.hi = []  # 小顶堆

    def addNum(self, num):
        heapq.heappush(self.lo, -num)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def findMedian(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2
```

### 2.6 O(1)插入删除随机 (LeetCode 380)

```python
import random

class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.pos = {}

    def insert(self, val):
        if val in self.pos: return False
        self.pos[val] = len(self.nums)
        self.nums.append(val)
        return True

    def remove(self, val):
        if val not in self.pos: return False
        idx, last = self.pos[val], self.nums[-1]
        self.nums[idx] = last
        self.pos[last] = idx
        self.nums.pop()
        del self.pos[val]
        return True

    def get_random(self):
        return random.choice(self.nums)
```

### 2.7 实现并查集 (LeetCode 684等)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]  # 路径压缩
            x = self.parent[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]: px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]: self.rank[px] += 1
        return True
```

---

## 三、复杂度分析

| 数据结构 | 操作 | 时间复杂度 |
|---------|------|-----------|
| LRU Cache | get/put | $O(1)$ |
| LFU Cache | get/put | $O(1)$ |
| Trie | insert/search | $O(m)$ m为单词长度 |
| MinStack | 所有操作 | $O(1)$ |
| MedianFinder | addNum | $O(\log n)$ |
| RandomizedSet | 所有操作 | $O(1)$ |
| UnionFind | find/union | $O(\alpha(n))$ |

---

## 四、面试高频题

1. **LeetCode 146：** LRU缓存
2. **LeetCode 460：** LFU缓存
3. **LeetCode 208：** 实现Trie
4. **LeetCode 155：** 最小栈
5. **LeetCode 295：** 数据流的中位数
6. **LeetCode 380：** O(1)时间插入删除随机
7. **LeetCode 225：** 用队列实现栈
8. **LeetCode 384：** 打乱数组
9. **LeetCode 706：** 设计哈希映射
10. **LeetCode 211：** 添加与搜索单词
