# 11-滑动窗口基础 (Sliding Window Basics)

滑动窗口是一种通过维护一个可变或固定大小的窗口来简化问题的方法，将嵌套循环优化为单循环。

## 固定窗口

```javascript
// 固定窗口大小 - 在数组上滑动固定长度的窗口
function fixedWindow(arr, k) {
  let windowSum = 0;
  for (let i = 0; i < k; i++) windowSum += arr[i];
  let maxSum = windowSum;

  for (let i = k; i < arr.length; i++) {
    windowSum += arr[i] - arr[i - k];
    maxSum = Math.max(maxSum, windowSum);
  }
  return maxSum;
}
```

## 可变窗口

```javascript
// 可变窗口大小 - 根据条件动态调整
function variableWindow(arr, target) {
  let left = 0, windowSum = 0, minLen = Infinity;
  for (let right = 0; right < arr.length; right++) {
    windowSum += arr[right];
    while (windowSum >= target) {
      minLen = Math.min(minLen, right - left + 1);
      windowSum -= arr[left++];
    }
  }
  return minLen === Infinity ? 0 : minLen;
}
```

## C++ 实现

```cpp
#include <vector>
#include <climits>
using namespace std;

// 固定窗口最大和
int maxSumFixed(vector<int>& nums, int k) {
    int window = 0;
    for (int i = 0; i < k; i++) window += nums[i];
    int maxSum = window;
    for (int i = k; i < nums.size(); i++) {
        window += nums[i] - nums[i - k];
        maxSum = max(maxSum, window);
    }
    return maxSum;
}

// 最短子数组和 >= target
int minSubArrayLen(int target, vector<int>& nums) {
    int left = 0, sum = 0, minLen = INT_MAX;
    for (int right = 0; right < nums.size(); right++) {
        sum += nums[right];
        while (sum >= target) {
            minLen = min(minLen, right - left + 1);
            sum -= nums[left++];
        }
    }
    return minLen == INT_MAX ? 0 : minLen;
}
```

## 通用滑动窗口模板

```javascript
function slidingWindow(s) {
  let left = 0, right = 0;
  let window = {};

  while (right < s.length) {
    // 扩大窗口
    let char = s[right];
    window[char] = (window[char] || 0) + 1;
    right++;

    // 满足条件时收缩
    while (needShrink(window)) {
      let d = s[left];
      window[d]--;
      if (window[d] === 0) delete window[d];
      left++;
    }
  }
}
```

## 典型例题

```javascript
// 无重复字符的最长子串
function lengthOfLongestSubstring(s) {
  let set = new Set();
  let left = 0, maxLen = 0;
  for (let right = 0; right < s.length; right++) {
    while (set.has(s[right])) {
      set.delete(s[left++]);
    }
    set.add(s[right]);
    maxLen = Math.max(maxLen, right - left + 1);
  }
  return maxLen;
}

// 包含所有字符的最短子串
function minWindow(s, t) {
  let need = {};
  for (let c of t) need[c] = (need[c] || 0) + 1;
  let count = Object.keys(need).length;
  let left = 0, minLen = Infinity, start = 0;
  let window = {};
  for (let right = 0; right < s.length; right++) {
    let c = s[right];
    window[c] = (window[c] || 0) + 1;
    if (need[c] && window[c] === need[c]) count--;
    while (count === 0) {
      if (right - left + 1 < minLen) {
        minLen = right - left + 1;
        start = left;
      }
      let d = s[left++];
      window[d]--;
      if (need[d] && window[d] < need[d]) count++;
    }
  }
  return minLen === Infinity ? "" : s.substr(start, minLen);
}
```

## 复杂度分析

| 方法 | 时间 | 空间 |
|------|------|------|
| 暴力枚举子数组 | O(n²) 或 O(n³) | O(1) |
| 滑动窗口 | O(n) | O(k) |

滑动窗口的关键：每个元素最多被 left 和 right 各访问一次，总操作 O(n)。

## 常见陷阱

1. **窗口收缩条件**：while 还是 if，需要仔细判断
2. **边界更新**：先更新窗口还是先记录答案
3. **空窗口处理**：left > right 时的情况
4. **字符计数**：用 Map 还是数组（ASCII 可用数组）
