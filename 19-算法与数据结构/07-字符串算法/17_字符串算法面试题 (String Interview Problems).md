# String Interview Problems


```javascript
字符串算法在面试中的高频题型。```


```
// 1. 无重复字符的最长字串
function lengthOfLongestSubstring(s) {
  const map = new Map(); let max = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    if (map.has(s[i])) start = Math.max(start, map.get(s[i])+1);
    map.set(s[i], i); max = Math.max(max, i-start+1);
  }
  return max;
}
// 2. 字符串的排列
function checkInclusion(s1, s2) {
  const need = new Map(); for (const c of s1) need.set(c,(need.get(c)||0)+1);
  const win = new Map(); let l=0, r=0, valid=0;
  while (r < s2.length) {
    const c = s2[r++];
    if (need.has(c)) { win.set(c,(win.get(c)||0)+1); if (win.get(c)===need.get(c)) valid++; }
    while (r - l >= s1.length) {
      if (valid === need.size) return true;
      const d = s2[l++];
      if (need.has(d)) { if (win.get(d)===need.get(d)) valid--; win.set(d,win.get(d)-1); }
    }
  }
  return false;
}
console.log(lengthOfLongestSubstring("abcabcbb")); // 3
console.log(checkInclusion("ab", "eidbaooo")); // true```


## 经典例题详解

### 3. 最长回文子串（中心扩展法）

```javascript
function longestPalindrome(s) {
  let start = 0, maxLen = 1;

  function expand(l, r) {
    while (l >= 0 && r < s.length && s[l] === s[r]) { l--; r++; }
    if (r - l - 1 > maxLen) { maxLen = r - l - 1; start = l + 1; }
  }

  for (let i = 0; i < s.length; i++) {
    expand(i, i);     // 奇数长度回文
    expand(i, i + 1); // 偶数长度回文
  }
  return s.slice(start, start + maxLen);
}
```

### 4. 最小覆盖子串（滑动窗口）

```javascript
// LeetCode 76
function minWindow(s, t) {
  const need = new Map();
  for (const c of t) need.set(c, (need.get(c) || 0) + 1);

  const win = new Map();
  let valid = 0, l = 0, r = 0;
  let minLen = Infinity, minStart = 0;

  while (r < s.length) {
    const c = s[r++];
    if (need.has(c)) {
      win.set(c, (win.get(c) || 0) + 1);
      if (win.get(c) === need.get(c)) valid++;
    }
    while (valid === need.size) {
      if (r - l < minLen) { minLen = r - l; minStart = l; }
      const d = s[l++];
      if (need.has(d)) {
        if (win.get(d) === need.get(d)) valid--;
        win.set(d, win.get(d) - 1);
      }
    }
  }
  return minLen === Infinity ? '' : s.slice(minStart, minStart + minLen);
}
```

### 5. 字符串转换整数 atoi

```javascript
function myAtoi(s) {
  s = s.trimStart();
  if (!s.length) return 0;
  let sign = 1, i = 0;
  if (s[0] === '-') { sign = -1; i++; }
  else if (s[0] === '+') { i++; }
  let result = 0;
  while (i < s.length && s[i] >= '0' && s[i] <= '9') {
    result = result * 10 + (+s[i++]);
    if (result * sign > 2147483647) return 2147483647;
    if (result * sign < -2147483648) return -2147483648;
  }
  return result * sign;
}
```

## 面试高频技巧总结

  | 技巧 | 适用场景 | 典型题目 |
  | --- | --- | --- |
  | 滑动窗口 | 连续子串/子数组 | 76, 3, 438 |
  | 双指针 | 回文、有序数组 | 5, 125 |
  | 哈希表 | 字符计数、去重 | 3, 383 |
  | KMP | 模式匹配 | 28, 459 |
  | 动态规划 | 编辑距离、LCS | 72, 1143 |
  | 前缀树 | 前缀匹配 | 208, 211 |

## 滑动窗口模板

```javascript
// 滑动窗口通用模板
function slidingWindow(s, condition) {
  const window = new Map();
  let left = 0, valid = 0;
  const result = [];

  for (let right = 0; right < s.length; right++) {
    const c = s[right];
    // 扩大窗口：更新窗口数据
    // window.set(c, ...)

    while (/* 窗口需要收缩 */) {
      const d = s[left++];
      // 缩小窗口：更新窗口数据
    }

    // 收集结果
  }
  return result;
}
```

  点击按钮查看结果
