# 37-算法中的滑动窗口技巧 (Sliding Window)

滑动窗口是数组/字符串子区间问题的核心技巧，通过维护一个可变大小的窗口在线性时间内解决问题。

## 核心思想

用两个指针 l, r 表示窗口 [l, r)，右指针扩大窗口，左指针缩小窗口，窗口内的状态用哈希表或计数器维护。

## 窗口类型

| 类型 | 特征 | 典型问题 |
|------|------|---------|
| 定长窗口 | 窗口大小固定 | 滑动平均、固定长度子数组 |
| 变长窗口（最小） | 满足条件时缩小 | 最小覆盖子串 |
| 变长窗口（最大） | 不满足条件时缩小 | 最长无重复子串 |

## JavaScript 实现

```javascript
// 通用滑动窗口框架
function slidingWindowTemplate(s) {
  const window = new Map();
  let l = 0, r = 0, valid = 0;
  let ans = /* 根据题意初始化 */;

  while (r < s.length) {
    // 扩大窗口
    const c = s[r++];
    window.set(c, (window.get(c) || 0) + 1);
    // 更新 valid 计数

    // 缩小窗口条件
    while (/* 需要缩小 */) {
      const d = s[l++];
      window.set(d, window.get(d) - 1);
      if (window.get(d) === 0) window.delete(d);
      // 更新 valid 计数
    }

    // 更新答案
  }
  return ans;
}

// 最小覆盖子串（LeetCode 76）
function minWindow(s, t) {
  const need = new Map();
  for (const c of t) need.set(c, (need.get(c) || 0) + 1);

  const window = new Map();
  let l = 0, r = 0, valid = 0;
  let start = 0, len = Infinity;

  while (r < s.length) {
    const c = s[r++];
    if (need.has(c)) {
      window.set(c, (window.get(c) || 0) + 1);
      if (window.get(c) === need.get(c)) valid++;
    }

    while (valid === need.size) {
      if (r - l < len) { start = l; len = r - l; }
      const d = s[l++];
      if (need.has(d)) {
        if (window.get(d) === need.get(d)) valid--;
        window.set(d, window.get(d) - 1);
      }
    }
  }
  return len === Infinity ? '' : s.substring(start, start + len);
}

// 最长无重复字符子串（LeetCode 3）
function lengthOfLongestSubstring(s) {
  const window = new Map();
  let l = 0, r = 0, maxLen = 0;

  while (r < s.length) {
    const c = s[r++];
    window.set(c, (window.get(c) || 0) + 1);

    while (window.get(c) > 1) {
      const d = s[l++];
      window.set(d, window.get(d) - 1);
    }
    maxLen = Math.max(maxLen, r - l);
  }
  return maxLen;
}

// 定长窗口：大小为 k 的最大平均值
function findMaxAverage(nums, k) {
  let sum = 0;
  for (let i = 0; i < k; i++) sum += nums[i];
  let maxSum = sum;
  for (let i = k; i < nums.length; i++) {
    sum += nums[i] - nums[i - k];
    maxSum = Math.max(maxSum, sum);
  }
  return maxSum / k;
}

// 测试
console.log(minWindow('ADOBECODEBANC', 'ABC'));  // BANC
console.log(lengthOfLongestSubstring('abcabcbb')); // 3
console.log(findMaxAverage([1, 12, -5, -6, 50, 3], 1)); // 12.0
```

## C++ 实现

```cpp
#include <string>
#include <unordered_map>
#include <algorithm>
using namespace std;

// 最长无重复字符子串
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> window;
    int l = 0, maxLen = 0;
    for (int r = 0; r < s.size(); r++) {
        window[s[r]]++;
        while (window[s[r]] > 1) {
            window[s[l]]--;
            l++;
        }
        maxLen = max(maxLen, r - l + 1);
    }
    return maxLen;
}

// 最小覆盖子串
string minWindow(string s, string t) {
    unordered_map<char, int> need, window;
    for (char c : t) need[c]++;
    int l = 0, valid = 0, start = 0, len = INT_MAX;
    for (int r = 0; r < s.size(); r++) {
        if (need.count(s[r])) {
            window[s[r]]++;
            if (window[s[r]] == need[s[r]]) valid++;
        }
        while (valid == need.size()) {
            if (r - l + 1 < len) { start = l; len = r - l + 1; }
            if (need.count(s[l])) {
                if (window[s[l]] == need[s[l]]) valid--;
                window[s[l]]--;
            }
            l++;
        }
    }
    return len == INT_MAX ? "" : s.substr(start, len);
}
```

## 复杂度

| 问题 | 时间 | 空间 |
|------|------|------|
| 最小覆盖子串 | O(n + m) | O(k) k为字符集 |
| 最长无重复子串 | O(n) | O(k) |
| 定长窗口最大值 | O(n) | O(1) |

## 常见陷阱

1. **窗口表示**：[l, r) 左闭右开最常用，注意 r 指向的是下一个待处理元素
2. **更新时机**：在扩大窗口后还是缩小窗口后更新答案，取决于求最小还是最大
3. **valid 计数**：只在 window[c] === need[c] 时加一，不是每次出现都加
4. **边界条件**：空字符串、t 比 s 长等特殊情况

## 实际应用

滑动窗口是面试中出现频率最高的技巧之一。LeetCode 76、3、438、567 等题目都是其经典应用。关键是识别问题是否具有"窗口单调性"——窗口扩大时某种条件单调变化。
