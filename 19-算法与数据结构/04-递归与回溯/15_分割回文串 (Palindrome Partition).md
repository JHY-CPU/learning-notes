# Palindrome Partition


```javascript
将字符串分割成若干回文子串，返回所有可能的分割方案。```

## 概念说明

给定字符串 s，将其分割成若干子串，使每个子串都是回文串，返回所有可能的分割方案。这是一个将回溯应用于字符串的经典问题，结合了回文判断和区间分割两个子问题。

## 核心思路

从字符串起始位置开始，尝试所有可能的分割点。对每段子串判断是否回文，若回文则递归处理剩余部分。回文判断使用双指针法，也可用动态规划预处理（DP[i][j] 表示 s[i..j] 是否回文）来优化为 O(1) 查询。

## 复杂度分析

- **时间复杂度：** O(n * 2^n)，最坏情况下 2^(n-1) 种分割方案，每种方案回文判断 O(n)。
- **空间复杂度：** O(n)，递归栈深度。若使用 DP 预处理，额外 O(n^2)。

## 适用场景

- 回文串相关分割问题
- 文本分词的暴力搜索
- 编辑器中的对称性检测

```
function partition(s) {
  const res = [];
  function isPalindrome(str, l, r) {
    while (l < r) if (str[l++] !== str[r--]) return false;
    return true;
  }
  function backtrack(start, path) {
    if (start === s.length) { res.push([...path]); return; }
    for (let i = start; i < s.length; i++) {
      if (isPalindrome(s, start, i)) {
        path.push(s.slice(start, i+1));
        backtrack(i+1, path);
        path.pop();
      }
    }
  }
  backtrack(0, []);
  return res;
}
console.log(partition('aab'));
// [["a","a","b"],["aa","b"]]```


## 常见变体与技巧

- **DP 预处理回文：** `dp[i][j] = s[i]===s[j] && dp[i+1][j-1]`，O(n^2) 预处理后回文判断 O(1)。
- **最少分割次数：** 改为求最少分割次数（LeetCode 132），用 DP `dp[i] = min(dp[j]+1)` 求解。
- **回文串计数：** 统计所有回文子串数量可用 Manacher 算法 O(n) 求解。

## DP 预处理优化

```javascript
function partitionDP(s) {
  const n = s.length;
  // 预处理回文表
  const isPal = Array.from({length: n}, () => new Array(n).fill(true));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) {
      isPal[i][j] = s[i] === s[j] && isPal[i+1][j-1];
    }
  }

  const res = [];
  function backtrack(start, path) {
    if (start === n) { res.push([...path]); return; }
    for (let i = start; i < n; i++) {
      if (isPal[start][i]) {  // O(1) 查询
        path.push(s.slice(start, i + 1));
        backtrack(i + 1, path);
        path.pop();
      }
    }
  }
  backtrack(0, []);
  return res;
}
```

## 最少分割次数（LeetCode 132）

```javascript
// 动态规划求最少分割次数
function minCut(s) {
  const n = s.length;
  const isPal = Array.from({length: n}, () => new Array(n).fill(false));
  for (let i = n - 1; i >= 0; i--)
    for (let j = i; j < n; j++)
      isPal[i][j] = s[i] === s[j] && (j - i < 2 || isPal[i+1][j-1]);

  const dp = new Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    if (isPal[0][i]) { dp[i] = 0; continue; }
    for (let j = 0; j < i; j++) {
      if (isPal[j+1][i]) dp[i] = Math.min(dp[i], dp[j] + 1);
    }
  }
  return dp[n - 1];
}
```

  点击按钮查看结果
