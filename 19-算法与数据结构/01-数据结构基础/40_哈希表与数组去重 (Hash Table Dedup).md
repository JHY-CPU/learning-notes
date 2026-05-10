# 哈希表与数组去重 (Hash Table Dedup)

## 一、去重的基本方法

### 1.1 利用哈希集合 (Set)

最直接的方法，时间 $O(n)$，空间 $O(n)$。

```python
def deduplicate(nums):
    return list(set(nums))

# 保留顺序的去重
def deduplicate_ordered(nums):
    seen = set()
    result = []
    for x in nums:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
```

### 1.2 利用哈希表记录频次

```python
from collections import Counter

def count_and_dedup(nums):
    freq = Counter(nums)
    return list(freq.keys()), dict(freq)
```

---

## 二、原地去重

### 2.1 有序数组去重 (LeetCode 26)

```python
def remove_duplicates(nums):
    if not nums: return 0
    slow = 0
    for fast in range(1, len(nums)):
        if nums[fast] != nums[slow]:
            slow += 1
            nums[slow] = nums[fast]
    return slow + 1
```

**C++ 实现：**

```cpp
int removeDuplicates(vector<int>& nums) {
    if (nums.empty()) return 0;
    int slow = 0;
    for (int fast = 1; fast < nums.size(); fast++) {
        if (nums[fast] != nums[slow])
            nums[++slow] = nums[fast];
    }
    return slow + 1;
}
```

### 2.2 有序数组去重II — 最多保留两个 (LeetCode 80)

```python
def remove_duplicates_ii(nums):
    if len(nums) <= 2: return len(nums)
    slow = 2
    for fast in range(2, len(nums)):
        if nums[fast] != nums[slow - 2]:
            nums[slow] = nums[fast]
            slow += 1
    return slow
```

---

## 三、哈希表去重的应用

### 3.1 判断重复元素 (LeetCode 217)

```python
def contains_duplicate(nums):
    return len(nums) != len(set(nums))
```

### 3.2 存在重复元素II — 下标差不超过k (LeetCode 219)

```python
def contains_nearby_duplicate(nums, k):
    seen = {}
    for i, num in enumerate(nums):
        if num in seen and i - seen[num] <= k:
            return True
        seen[num] = i
    return False
```

### 3.3 存在重复元素III — 值差不超过t (LeetCode 220)

```python
def contains_nearby_almost_duplicate(nums, index_diff, value_diff):
    if value_diff < 0: return False
    bucket = {}
    w = value_diff + 1

    for i, num in enumerate(nums):
        key = num // w
        if key in bucket: return True
        if key - 1 in bucket and abs(num - bucket[key-1]) < w: return True
        if key + 1 in bucket and abs(num - bucket[key+1]) < w: return True
        bucket[key] = num
        if i >= index_diff:
            del bucket[nums[i - index_diff] // w]

    return False
```

---

## 四、复杂度分析

| 操作 | 时间 | 空间 |
|------|------|------|
| Set去重 | $O(n)$ | $O(n)$ |
| 有序原地去重 | $O(n)$ | $O(1)$ |
| 哈希判重 | $O(n)$ | $O(n)$ |
| 滑动窗口+桶 | $O(n)$ | $O(k)$ |

---

## 五、面试要点

1. **有序 vs 无序** — 决定能否用双指针
2. **是否保留顺序** — set 不保序
3. **原地 vs 额外空间** — 双指针可原地
4. **重复次数限制** — 最多保留k个
