# 43-哈希表与连续序列 (Hash Table Sequence)

利用哈希集合的 O(1) 查找特性，可以在 O(n) 时间内找到数组中的最长连续序列。

## 最长连续序列

```javascript
// 核心技巧：只从序列起点开始计数，避免重复计算
function longestConsecutive(nums) {
  const set = new Set(nums);
  let longest = 0;

  for (const n of set) {
    // n-1 不在集合中时，n 是某个序列的起点
    if (!set.has(n - 1)) {
      let len = 1;
      while (set.has(n + len)) len++;
      longest = Math.max(longest, len);
    }
  }
  return longest;
}

console.log(longestConsecutive([100, 4, 200, 1, 3, 2])); // 4
```

## C++ 实现

```cpp
#include <vector>
#include <unordered_set>
#include <algorithm>
using namespace std;

int longestConsecutive(vector<int>& nums) {
    unordered_set<int> s(nums.begin(), nums.end());
    int longest = 0;
    for (int n : s) {
        if (!s.count(n - 1)) { // 序列起点
            int len = 1;
            while (s.count(n + len)) len++;
            longest = max(longest, len);
        }
    }
    return longest;
}
```

## 排序法（对比）

```javascript
// O(n log n) 方法：排序后扫描
function longestConsecutiveSort(nums) {
  if (!nums.length) return 0;
  nums.sort((a, b) => a - b);
  let longest = 1, curr = 1;
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] === nums[i - 1]) continue; // 跳过重复
    if (nums[i] === nums[i - 1] + 1) curr++;
    else curr = 1;
    longest = Math.max(longest, curr);
  }
  return longest;
}
```

## 方法对比

| 方法 | 时间 | 空间 |
|------|------|------|
| 排序法 | O(n log n) | O(1) |
| 哈希集合法 | O(n) | O(n) |
| 并查集法 | O(n * alpha(n)) | O(n) |

每个元素在哈希集合法中最多被访问两次（一次判断起点，一次在序列中计数），所以总时间 O(n)。

## 连续子数组问题

```javascript
// 连续子数组和是否为 k 的倍数
function checkSubarraySum(nums, k) {
  const map = new Map();
  map.set(0, -1); // 前缀和 0 出现在索引 -1
  let sum = 0;

  for (let i = 0; i < nums.length; i++) {
    sum += nums[i];
    const mod = ((sum % k) + k) % k;
    if (map.has(mod)) {
      if (i - map.get(mod) >= 2) return true; // 子数组长度 >= 2
    } else {
      map.set(mod, i);
    }
  }
  return false;
}

// 最长连续递增子序列
function findLengthOfLCIS(nums) {
  let maxLen = 1, curr = 1;
  for (let i = 1; i < nums.length; i++) {
    if (nums[i] > nums[i - 1]) curr++;
    else curr = 1;
    maxLen = Math.max(maxLen, curr);
  }
  return maxLen;
}
```

## 丢失的第一个正数

```javascript
// 用哈希集合在 O(n) 时间和空间内解决
function firstMissingPositive(nums) {
  const set = new Set(nums);
  let i = 1;
  while (set.has(i)) i++;
  return i;
}

// 原地哈希 O(n) 时间 O(1) 空间
function firstMissingPositiveO1(nums) {
  const n = nums.length;
  for (let i = 0; i < n; i++) {
    while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] !== nums[i]) {
      [nums[nums[i] - 1], nums[i]] = [nums[i], nums[nums[i] - 1]];
    }
  }
  for (let i = 0; i < n; i++) {
    if (nums[i] !== i + 1) return i + 1;
  }
  return n + 1;
}
```

## 常见陷阱

1. **起点判断**：必须检查 `n-1` 不在集合中才开始计数
2. **重复元素**：排序法中要跳过重复
3. **空数组**：输入为空时返回 0
4. **全相同元素**：最长连续序列长度为 1
5. **负数序列**：算法对负数同样有效
