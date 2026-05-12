# 12-滑动窗口进阶 (Sliding Window Advanced)

进阶滑动窗口技巧，包括多条件窗口、双哈希表维护、计数窗口等复杂场景。

## 字符串排列判断

```javascript
// s2 中是否包含 s1 的排列（字母异位词）
function checkInclusion(s1, s2) {
  let need = {}, window = {};
  for (let c of s1) need[c] = (need[c] || 0) + 1;

  let left = 0, right = 0, valid = 0;
  while (right < s2.length) {
    let c = s2[right++];
    if (need[c]) {
      window[c] = (window[c] || 0) + 1;
      if (window[c] === need[c]) valid++;
    }

    while (right - left >= s1.length) {
      if (valid === Object.keys(need).length) return true;
      let d = s2[left++];
      if (need[d]) {
        if (window[d] === need[d]) valid--;
        window[d]--;
      }
    }
  }
  return false;
}
```

## C++ 实现

```cpp
#include <string>
#include <unordered_map>
using namespace std;

bool checkInclusion(string s1, string s2) {
    unordered_map<char, int> need, window;
    for (char c : s1) need[c]++;
    int left = 0, right = 0, valid = 0;

    while (right < s2.size()) {
        char c = s2[right++];
        if (need.count(c)) {
            window[c]++;
            if (window[c] == need[c]) valid++;
        }
        while (right - left >= s1.size()) {
            if (valid == need.size()) return true;
            char d = s2[left++];
            if (need.count(d)) {
                if (window[d] == need[d]) valid--;
                window[d]--;
            }
        }
    }
    return false;
}
```

## 找所有字母异位词

```javascript
function findAnagrams(s, p) {
  let need = {}, window = {};
  for (let c of p) need[c] = (need[c] || 0) + 1;

  let left = 0, right = 0, valid = 0;
  let result = [];

  while (right < s.length) {
    let c = s[right++];
    if (need[c]) {
      window[c] = (window[c] || 0) + 1;
      if (window[c] === need[c]) valid++;
    }

    while (right - left >= p.length) {
      if (valid === Object.keys(need).length) result.push(left);
      let d = s[left++];
      if (need[d]) {
        if (window[d] === need[d]) valid--;
        window[d]--;
      }
    }
  }
  return result;
}
```

## 最小覆盖子串

```javascript
function minWindow(s, t) {
  let need = {};
  for (let c of t) need[c] = (need[c] || 0) + 1;
  let needCount = Object.keys(need).length;

  let window = {};
  let left = 0, valid = 0;
  let minLen = Infinity, start = 0;

  for (let right = 0; right < s.length; right++) {
    let c = s[right];
    window[c] = (window[c] || 0) + 1;
    if (need[c] && window[c] === need[c]) valid++;

    while (valid === needCount) {
      if (right - left + 1 < minLen) {
        minLen = right - left + 1;
        start = left;
      }
      let d = s[left++];
      window[d]--;
      if (need[d] && window[d] < need[d]) valid--;
    }
  }
  return minLen === Infinity ? "" : s.slice(start, start + minLen);
}
```

## K 个不同字符的最长子串

```javascript
function longestKDistinct(s, k) {
  let count = {};
  let left = 0, maxLen = 0;
  for (let right = 0; right < s.length; right++) {
    count[s[right]] = (count[s[right]] || 0) + 1;
    while (Object.keys(count).length > k) {
      count[s[left]]--;
      if (count[s[left]] === 0) delete count[s[left]];
      left++;
    }
    maxLen = Math.max(maxLen, right - left + 1);
  }
  return maxLen;
}
```

## 滑动窗口最大值（单调队列）

```javascript
function maxSlidingWindow(nums, k) {
  let deque = []; // 存索引，对应值单调递减
  let result = [];

  for (let i = 0; i < nums.length; i++) {
    // 移除超出窗口的元素
    while (deque.length && deque[0] <= i - k) deque.shift();
    // 维护单调递减
    while (deque.length && nums[deque[deque.length - 1]] < nums[i]) deque.pop();
    deque.push(i);
    if (i >= k - 1) result.push(nums[deque[0]]);
  }
  return result;
}
```

## 进阶技巧总结

1. **valid 计数器**：用 valid 变量跟踪窗口中满足条件的字符种类数
2. **双哈希表**：need 和 window 分别记录需求和当前状态
3. **窗口收缩时机**：固定长度窗口用 `right - left >= len`，不定长用 `valid === needCount`
4. **答案记录位置**：在收缩窗口时记录（找最小）或在满足条件时记录（找最大）
