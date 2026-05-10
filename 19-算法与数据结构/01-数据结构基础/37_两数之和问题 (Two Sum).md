# 两数之和问题 (Two Sum)

## 一、问题描述

给定一个整数数组 `nums` 和目标值 `target`，找出数组中和为目标值的两个整数的下标。

**LeetCode 1 — 入门必刷题**

---

## 二、解法分析

### 2.1 暴力法 — O(n^2)

双重循环，检查每对元素：

```python
def two_sum_brute(nums, target):
    n = len(nums)
    for i in range(n):
        for j in range(i + 1, n):
            if nums[i] + nums[j] == target:
                return [i, j]
    return []
```

**时间复杂度：** $O(n^2)$
**空间复杂度：** $O(1)$

### 2.2 哈希表 — O(n)

遍历时用哈希表记录已见过的数，查找 complement 是否存在：

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

**时间复杂度：** $O(n)$
**空间复杂度：** $O(n)$

### 2.3 排序+双指针 — O(n log n)

先排序，然后用双指针从两端向中间逼近：

```python
def two_sum_sorted(nums, target):
    # 需要保留原始下标
    indexed = sorted(enumerate(nums), key=lambda x: x[1])
    left, right = 0, len(indexed) - 1

    while left < right:
        s = indexed[left][1] + indexed[right][1]
        if s == target:
            return [indexed[left][0], indexed[right][0]]
        elif s < target:
            left += 1
        else:
            right -= 1
    return []
```

**时间复杂度：** $O(n \log n)$
**空间复杂度：** $O(n)$

---

## 三、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 哈希表解法
vector<int> twoSum(vector<int>& nums, int target) {
    unordered_map<int, int> seen;
    for (int i = 0; i < nums.size(); i++) {
        int complement = target - nums[i];
        if (seen.count(complement)) {
            return {seen[complement], i};
        }
        seen[nums[i]] = i;
    }
    return {};
}
```

---

## 四、变种问题

### 4.1 返回所有配对

```python
def two_sum_all(nums, target):
    seen = {}
    result = []
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            result.append([seen[complement], i])
        seen[num] = i
    return result
```

### 4.2 有序数组 (LeetCode 167)

```python
def two_sum_sorted_only(numbers, target):
    left, right = 0, len(numbers) - 1
    while left < right:
        s = numbers[left] + numbers[right]
        if s == target:
            return [left + 1, right + 1]  # 1-indexed
        elif s < target:
            left += 1
        else:
            right -= 1
```

### 4.3 统计满足条件的对数

```python
def count_pairs(nums, target):
    from collections import Counter
    count = Counter(nums)
    result = 0
    for x in count:
        y = target - x
        if y in count:
            if x == y:
                result += count[x] * (count[x] - 1) // 2
            elif x < y:
                result += count[x] * count[y]
    return result
```

---

## 五、复杂度总结

| 方法 | 时间 | 空间 | 适用场景 |
|------|------|------|---------|
| 暴力 | $O(n^2)$ | $O(1)$ | 小数据 |
| 哈希表 | $O(n)$ | $O(n)$ | 通用首选 |
| 排序+双指针 | $O(n \log n)$ | $O(1)$ | 有序/只需值 |

---

## 六、面试要点

1. **首选哈希表方案** — 时间最优，代码简洁
2. **主动分析复杂度** — 面试官最关心
3. **处理边界** — 空数组、无解、重复元素
4. **举一反三** — 两数之和 → 三数之和 → K数之和
