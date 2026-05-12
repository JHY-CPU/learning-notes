# 49-回文串判断 (Palindrome)

回文串是正读反读都相同的字符串，双指针法 O(n) 判断。

## 双指针法

```javascript
// 基础回文判断
function isPalindrome(s) {
  s = s.replace(/[^a-zA-Z0-9]/g, '').toLowerCase();
  let l = 0, r = s.length - 1;
  while (l < r) {
    if (s[l] !== s[r]) return false;
    l++; r--;
  }
  return true;
}

console.log(isPalindrome("A man, a plan, a canal: Panama")); // true
console.log(isPalindrome("race a car")); // false
```

## C++ 实现

```cpp
#include <string>
using namespace std;

bool isPalindrome(string s) {
    int l = 0, r = s.size() - 1;
    while (l < r) {
        while (l < r && !isalnum(s[l])) l++;
        while (l < r && !isalnum(s[r])) r--;
        if (tolower(s[l++]) != tolower(s[r--])) return false;
    }
    return true;
}
```

## 最长回文子串

```javascript
// 中心扩展法 O(n²)
function longestPalindrome(s) {
  let start = 0, maxLen = 1;

  function expand(l, r) {
    while (l >= 0 && r < s.length && s[l] === s[r]) { l--; r++; }
    return r - l - 1;
  }

  for (let i = 0; i < s.length; i++) {
    const len1 = expand(i, i);     // 奇数长度回文
    const len2 = expand(i, i + 1); // 偶数长度回文
    const len = Math.max(len1, len2);
    if (len > maxLen) {
      maxLen = len;
      start = i - Math.floor((len - 1) / 2);
    }
  }
  return s.substring(start, start + maxLen);
}

// Manacher 算法 O(n)
function longestPalindromeManacher(s) {
  // 预处理：插入分隔符
  const t = '#' + s.split('').join('#') + '#';
  const n = t.length;
  const p = new Array(n).fill(0); // p[i] = 以 i 为中心的回文半径
  let center = 0, right = 0;

  for (let i = 0; i < n; i++) {
    if (i < right) p[i] = Math.min(right - i, p[2 * center - i]);
    while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n && t[i - p[i] - 1] === t[i + p[i] + 1]) {
      p[i]++;
    }
    if (i + p[i] > right) { center = i; right = i + p[i]; }
  }

  let maxLen = 0, maxCenter = 0;
  for (let i = 0; i < n; i++) {
    if (p[i] > maxLen) { maxLen = p[i]; maxCenter = i; }
  }
  const start = (maxCenter - maxLen) / 2;
  return s.substring(start, start + maxLen);
}
```

## 最长回文子序列

```javascript
// DP 解法 O(n²)
function longestPalindromeSubseq(s) {
  const n = s.length;
  const dp = Array.from({length: n}, () => new Array(n).fill(0));

  for (let i = n - 1; i >= 0; i--) {
    dp[i][i] = 1;
    for (let j = i + 1; j < n; j++) {
      if (s[i] === s[j]) dp[i][j] = dp[i+1][j-1] + 2;
      else dp[i][j] = Math.max(dp[i+1][j], dp[i][j-1]);
    }
  }
  return dp[0][n-1];
}
```

## 回文链表

```javascript
function isPalindromeList(head) {
  // 找中点
  let slow = head, fast = head;
  while (fast && fast.next) {
    slow = slow.next;
    fast = fast.next.next;
  }
  // 反转后半部分
  let prev = null;
  while (slow) {
    let next = slow.next;
    slow.next = prev;
    prev = slow;
    slow = next;
  }
  // 比较
  let p1 = head, p2 = prev;
  while (p2) {
    if (p1.val !== p2.val) return false;
    p1 = p1.next; p2 = p2.next;
  }
  return true;
}
```

## 复杂度分析

| 问题 | 方法 | 时间 | 空间 |
|------|------|------|------|
| 回文判断 | 双指针 | O(n) | O(1) |
| 最长回文子串 | 中心扩展 | O(n²) | O(1) |
| 最长回文子串 | Manacher | O(n) | O(n) |
| 最长回文子序列 | DP | O(n²) | O(n²) |
| 回文链表 | 反转比较 | O(n) | O(1) |

## 常见陷阱

1. **大小写和标点**：题目通常要求忽略大小写和非字母字符
2. **空字符串**：空字符串是回文
3. **单字符**：单个字符是回文
4. **中心扩展的奇偶**：要分别处理奇数和偶数长度的回文
