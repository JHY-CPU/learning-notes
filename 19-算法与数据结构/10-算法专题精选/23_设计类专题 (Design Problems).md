# 设计类专题 (Design Problems)

## 一、概念定义与原理

设计类题目要求实现具有特定功能的数据结构或类。重点在于选择合适的数据结构来满足时间/空间要求。

---

## 二、经典设计问题

### 2.1 LRU 缓存

**要求：** $O(1)$ 时间的 get 和 put 操作。

**方案：** 哈希表 + 双向链表。哈希表提供 $O(1)$ 查找，双向链表维护访问顺序。

### 2.2 数据流中位数

**要求：** 支持添加数字和查询中位数。

**方案：** 两个堆（大顶堆存较小的一半，小顶堆存较大的一半）。

### 2.3 最小栈

**要求：** $O(1)$ 获取栈中最小值。

**方案：** 辅助栈记录每个状态的最小值。

---

## 三、代码实现

### 3.1 LRU 缓存 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

class LRUCache {
    int capacity;
    list<pair<int,int>> cache; // {key, value}
    unordered_map<int, list<pair<int,int>>::iterator> map_;
public:
    LRUCache(int cap) : capacity(cap) {}

    int get(int key) {
        if (!map_.count(key)) return -1;
        // 移到链表头部
        auto it = map_[key];
        int val = it->second;
        cache.erase(it);
        cache.push_front({key, val});
        map_[key] = cache.begin();
        return val;
    }

    void put(int key, int value) {
        if (map_.count(key)) cache.erase(map_[key]);
        else if (cache.size() == capacity) {
            auto last = cache.back();
            map_.erase(last.first);
            cache.pop_back();
        }
        cache.push_front({key, value});
        map_[key] = cache.begin();
    }
};
```

### 3.2 数据流中位数 - C++

```cpp
class MedianFinder {
    priority_queue<int> left; // 大顶堆（较小的一半）
    priority_queue<int, vector<int>, greater<int>> right; // 小顶堆（较大的一半）
public:
    void addNum(int num) {
        if (left.empty() || num <= left.top()) left.push(num);
        else right.push(num);
        // 平衡两个堆
        if (left.size() > right.size() + 1) {
            right.push(left.top()); left.pop();
        }
        if (right.size() > left.size()) {
            left.push(right.top()); right.pop();
        }
    }
    double findMedian() {
        if (left.size() > right.size()) return left.top();
        return (left.top() + right.top()) / 2.0;
    }
};
```

### 3.3 Python 实现

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache: self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

import heapq
class MedianFinder:
    def __init__(self):
        self.left = []   # 大顶堆（取负）
        self.right = []  # 小顶堆
    def addNum(self, num):
        heapq.heappush(self.left, -num)
        heapq.heappush(self.right, -heapq.heappop(self.left))
        if len(self.right) > len(self.left):
            heapq.heappush(self.left, -heapq.heappop(self.right))
    def findMedian(self):
        if len(self.left) > len(self.right): return -self.left[0]
        return (-self.left[0] + self.right[0]) / 2

# 测试
lru = LRUCache(2)
lru.put(1, 1); lru.put(2, 2)
print(lru.get(1))  # 1
lru.put(3, 3)      # 淘汰key=2
print(lru.get(2))  # -1
```

### 3.4 O(1) 插入删除随机获取

```cpp
class RandomizedSet {
    vector<int> nums;
    unordered_map<int, int> pos;
    mt19937 rng;
public:
    bool insert(int val) {
        if (pos.count(val)) return false;
        pos[val] = nums.size();
        nums.push_back(val);
        return true;
    }
    bool remove(int val) {
        if (!pos.count(val)) return false;
        int last = nums.back();
        pos[last] = pos[val];
        nums[pos[val]] = last;
        nums.pop_back();
        pos.erase(val);
        return true;
    }
    int getRandom() {
        return nums[uniform_int_distribution<int>(0, nums.size()-1)(rng)];
    }
};
```

---

## 四、复杂度分析

| 数据结构 | Get | Put/Insert | 其他操作 |
|---------|-----|-----------|---------|
| LRU 缓存 | $O(1)$ | $O(1)$ | |
| 中位数 | $O(1)$ | $O(\log n)$ | |
| 最小栈 | $O(1)$ | $O(1)$ | |
| RandomizedSet | $O(1)$ | $O(1)$ | Random $O(1)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 146：** LRU 缓存
2. **LeetCode 295：** 数据流的中位数
3. **LeetCode 155：** 最小栈
4. **LeetCode 380：** O(1) 时间插入、删除和获取随机元素
5. **LeetCode 460：** LFU 缓存
