# 哈希表专题 (Hash Table Problems)

## 一、概念定义与原理

### 1.1 哈希表基础

哈希表通过哈希函数将键映射到数组下标，实现平均 $O(1)$ 的查找、插入和删除。

**核心概念：**
- **哈希函数：** 将任意大小的数据映射到固定大小的值
- **桶 (Bucket)：** 存储元素的数组位置
- **碰撞 (Collision)：** 不同的键映射到同一个桶

### 1.2 碰撞处理方法

| 方法 | 原理 | 优缺点 |
|------|------|--------|
| 链地址法 | 每个桶存链表 | 实现简单，最坏 $O(n)$ |
| 开放地址法 | 冲突时探测下一个空位 | 聚集效应，负载因子要求低 |
| 双重哈希 | 用第二个哈希函数计算步长 | 减少聚集，计算开销大 |

### 1.3 时间复杂度

| 操作 | 平均 | 最坏 |
|------|------|------|
| 查找 | $O(1)$ | $O(n)$ |
| 插入 | $O(1)$ | $O(n)$ |
| 删除 | $O(1)$ | $O(n)$ |

---

## 二、核心技巧

### 2.1 哈希计数

用哈希表统计频次，是最常见的应用模式。

```python
from collections import Counter

def most_frequent(nums):
    count = Counter(nums)
    return max(count, key=count.get)

# 字母异位词判断 (LeetCode 242)
def is_anagram(s, t):
    return Counter(s) == Counter(t)
```

### 2.2 哈希映射存储索引

```python
# 两数之和 (LeetCode 1)
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```

### 2.3 哈希分组

```python
# 字母异位词分组 (LeetCode 49)
def group_anagrams(strs):
    groups = {}
    for s in strs:
        key = ''.join(sorted(s))
        groups.setdefault(key, []).append(s)
    return list(groups.values())
```

### 2.4 前缀和 + 哈希

```python
# 和为K的子数组 (LeetCode 560)
def subarray_sum(nums, k):
    count, prefix_sum, seen = 0, 0, {0: 1}
    for num in nums:
        prefix_sum += num
        count += seen.get(prefix_sum - k, 0)
        seen[prefix_sum] = seen.get(prefix_sum, 0) + 1
    return count
```

---

## 三、经典题目详解

### 3.1 最长连续序列 (LeetCode 128)

```python
def longest_consecutive(nums):
    num_set = set(nums)
    best = 0
    for num in num_set:
        if num - 1 not in num_set:  # 只从序列起点开始
            curr = num
            length = 1
            while curr + 1 in num_set:
                curr += 1
                length += 1
            best = max(best, length)
    return best
```

**时间复杂度：** $O(n)$ — 每个元素最多被访问两次。

### 3.2 LRU缓存 (LeetCode 146)

```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.cap = capacity

    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.cap:
            self.cache.popitem(last=False)
```

### 3.3 常数时间插入删除随机 (LeetCode 380)

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

### 3.4 赎金信 (LeetCode 383)

```python
def can_construct(ransomNote, magazine):
    from collections import Counter
    mag_count = Counter(magazine)
    for c in ransomNote:
        if mag_count[c] <= 0:
            return False
        mag_count[c] -= 1
    return True
```

---

## 四、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 两数之和
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement))
            return {seen[complement], i};
        seen[nums[i]] = i;
    }
    return {};
}

// 最长连续序列
int longestConsecutive(vector<int>& nums) {
    unordered_set<int> numSet(nums.begin(), nums.end());
    int best = 0;
    for (int num : numSet) {
        if (!numSet.count(num - 1)) {
            int curr = num, len = 1;
            while (numSet.count(curr + 1)) { curr++; len++; }
            best = max(best, len);
        }
    }
    return best;
}

// 字母异位词分组
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> groups;
    for (string& s : strs) {
        string key = s;
        sort(key.begin(), key.end());
        groups[key].push_back(s);
    }
    vector<vector<string>> result;
    for (auto& [_, v] : groups) result.push_back(v);
    return result;
}
```

---

## 五、复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 两数之和 | $O(n)$ | $O(n)$ |
| 最长连续序列 | $O(n)$ | $O(n)$ |
| 字母异位词分组 | $O(nk\log k)$ | $O(nk)$ |
| 和为K的子数组 | $O(n)$ | $O(n)$ |
| LRU get/put | $O(1)$ | $O(cap)$ |

---

## 六、面试高频题

1. **LeetCode 1：** 两数之和
2. **LeetCode 49：** 字母异位词分组
3. **LeetCode 128：** 最长连续序列
4. **LeetCode 146：** LRU缓存
5. **LeetCode 380：** O(1)时间插入删除随机
6. **LeetCode 560：** 和为K的子数组
7. **LeetCode 242：** 有效的字母异位词
8. **LeetCode 383：** 赎金信
9. **LeetCode 205：** 同构字符串
10. **LeetCode 76：** 最小覆盖子串
