# Manacher


```javascript
Manacher 算法 O(n) 时间找出最长回文子串。```

## 概念说明

Manacher 算法是求最长回文子串的线性时间算法。核心思想是在已知回文的对称性质上加速判断：以 center 为中心的回文右边界为 right，对于 right 以内的位置 i，可利用其对称点 `2*center-i` 的已知回文半径来初始化 p[i]，从而避免重复比较。

## 核心思路

1. **预处理：** 在字符间插入 `#` 统一奇偶，如 `aba` 变为 `#a#b#a#`。
2. **维护窗口：** 用 center 和 right 记录当前最右回文的中心和右边界。
3. **对称性加速：** 若 i < right，`p[i] = min(right-i, p[2*center-i])` 利用对称信息。
4. **中心扩展：** 在初始化基础上向两侧扩展，更新 p[i]。
5. **更新边界：** 若 `i + p[i] > right`，更新 center 和 right。

## 复杂度分析

- **时间复杂度：** O(n)，right 指针只向右移动，每个字符最多被访问常数次。
- **空间复杂度：** O(n)，用于存储预处理字符串和回文半径数组 p。

## 适用场景

- 最长回文子串查询
- 回文串相关计数问题
- 字符串中的对称性分析

```
function manacher(s) {
  const t = '#' + s.split('').join('#') + '#';
  const p = new Array(t.length).fill(0);
  let center = 0, right = 0;
  for (let i = 0; i < t.length; i++) {
    if (i < right) p[i] = Math.min(right - i, p[2*center - i]);
    while (i-p[i]-1 >= 0 && i+p[i]+1 < t.length && t[i-p[i]-1] === t[i+p[i]+1]) p[i]++;
    if (i + p[i] > right) { center = i; right = i + p[i]; }
  }
  let maxLen = 0, centerIdx = 0;
  for (let i = 0; i < t.length; i++) {
    if (p[i] > maxLen) { maxLen = p[i]; centerIdx = i; }
  }
  const start = (centerIdx - maxLen) / 2;
  return s.slice(start, start + maxLen);
}
console.log(manacher("babad")); // "bab" 或 "aba"
console.log(manacher("cbbd")); // "bb" ```


## 常见变体与技巧

- **统计所有回文子串数量：** `sum(p[i])` 即为回文子串总数（利用 p 数组的回文半径信息）。
- **最长回文前缀/后缀：** 构造 `s + '#' + reverse(s)` 对其应用 Manacher。
- **对比中心扩展法：** 中心扩展法 O(n^2) 简单直观，Manacher 在 n 较大时优势明显。

  点击按钮查看结果
