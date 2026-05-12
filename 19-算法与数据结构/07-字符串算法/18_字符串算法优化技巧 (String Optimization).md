# String Optimization


```javascript
字符串算法的优化技巧包括预处理、双数组Trie、滚动哈希优化等。```

## 概念说明

字符串算法的优化通常从三个方面入手：减少重复计算（预处理）、使用高效数据结构（Trie、AC自动机）、利用数学技巧（滚动哈希、Z算法）。本节重点介绍 Z 算法和双数组 Trie 两种高级优化技巧。

## Z 算法说明

Z 算法在 O(n) 时间内计算 Z 数组，其中 Z[i] 表示字符串 s 与 s[i..] 的最长公共前缀长度。利用与 Manacher 类似的"窗口"技巧避免重复比较。应用：构造 `pattern + '$' + text`，Z 数组中值为 pattern.length 的位置即为匹配点。

## 双数组 Trie

Double-Array Trie 用 base 和 check 两个整数数组实现 Trie 树。查询时只需 `base[state] + code` 一步跳转，O(1) 时间完成单字符转移。相比普通 Trie 树大幅减少内存占用，适合大规模字典场景。

## 复杂度分析

- **Z 算法：** O(n) 时间，O(n) 空间
- **双数组 Trie：** 插入 O(len)，查询 O(len)，内存比普通 Trie 节省约 60%

## 适用场景

- 多模式匹配（Z算法 + 双数组Trie）
- 字典树密集查询场景（双数组Trie）
- 字符串快速匹配预处理（Z算法）

```
// 双数组 Trie（Double-Array Trie）
// 使用 base 和 check 两个数组实现，查询 O(1)
// 优势：内存紧凑、查询快速
//
// Z 算法（线性时间模式匹配预处理）
function zAlgorithm(s) {
  const n = s.length, z = new Array(n).fill(0);
  let l = 0, r = 0;
  for (let i = 1; i < n; i++) {
    if (i <= r) z[i] = Math.min(r-i+1, z[i-l]);
    while (i+z[i] < n && s[z[i]] === s[i+z[i]]) z[i]++;
    if (i+z[i]-1 > r) { l = i; r = i+z[i]-1; }
  }
  return z;
}
console.log(zAlgorithm("aaabaaaab")); // [0,2,1,0,3,4,2,1,0]
// 应用：字符串匹配时构造 s+$+t，计算 Z 数组```


## 常见变体与技巧

- **滚动哈希：** 用于 Rabin-Karp 算法，`hash(s[i..i+m]) = (hash(s[i-1..i+m-1]) - s[i-1]*base^(m-1)) * base + s[i+m]`。
- **后缀自动机（SAM）：** O(n) 构建，支持子串判断、不同子串计数、最长公共子串等高级应用。
- **工程实践：** 大规模文本搜索常将 KMP/AC自动机与 Bloom Filter 结合，先过滤再精确匹配。

  点击按钮查看结果
