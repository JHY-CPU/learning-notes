# Suffix Array Implementation


```javascript
后缀数组实现及其应用：最长公共前缀（LCP）、子串查询等。```


```
// 后缀数组（简化实现）
function buildSuffixArray(s) {
  const n = s.length;
  const suffixes = [];
  for (let i = 0; i < n; i++) suffixes.push({index: i, suffix: s.slice(i)});
  suffixes.sort((a,b) => a.suffix.localeCompare(b.suffix));
  return suffixes.map(s => s.index);
}
// LCP 数组（Kasai 算法）
function buildLCP(s, sa) {
  const n = s.length;
  const rank = new Array(n);
  for (let i = 0; i < n; i++) rank[sa[i]] = i;
  let k = 0;
  const lcp = new Array(n-1).fill(0);
  for (let i = 0; i < n; i++) {
    if (rank[i] === n-1) { k = 0; continue; }
    let j = sa[rank[i] + 1];
    while (i + k < n && j + k < n && s[i+k] === s[j+k]) k++;
    lcp[rank[i]] = k;
    if (k) k--;
  }
  return lcp;
}
const s = "banana";
const sa = buildSuffixArray(s);
console.log(sa); // [5,3,1,0,4,2] (a,ana,anana,banana,na,nana)```


## O(n log n) 倍增法构建

```javascript
// 倍增法构建后缀数组 O(n log n)
function buildSuffixArrayFast(s) {
  const n = s.length;
  let rank = Array.from({length: n}, (_, i) => s.charCodeAt(i));
  const sa = Array.from({length: n}, (_, i) => i);
  const tmp = new Array(n);

  for (let k = 1; k < n; k *= 2) {
    sa.sort((a, b) => {
      if (rank[a] !== rank[b]) return rank[a] - rank[b];
      const ra = a + k < n ? rank[a + k] : -1;
      const rb = b + k < n ? rank[b + k] : -1;
      return ra - rb;
    });

    tmp[sa[0]] = 0;
    for (let i = 1; i < n; i++) {
      tmp[sa[i]] = tmp[sa[i - 1]] +
        (rank[sa[i]] !== rank[sa[i - 1]] ||
         ((sa[i] + k < n ? rank[sa[i] + k] : -1) !==
          (sa[i - 1] + k < n ? rank[sa[i - 1] + k] : -1)) ? 1 : 0);
    }
    rank = [...tmp];
    if (rank[sa[n - 1]] === n - 1) break;  // 所有排名不同，提前终止
  }
  return sa;
}
```

## 后缀数组应用

```javascript
// 1. 子串查询：在排序后的后缀数组上二分查找
function searchSubstring(text, pattern) {
  const sa = buildSuffixArray(text);
  let lo = 0, hi = sa.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const suffix = text.slice(sa[mid]);
    if (suffix.startsWith(pattern)) return sa[mid];
    if (suffix < pattern) lo = mid + 1;
    else hi = mid - 1;
  }
  return -1;  // 未找到
}

// 2. 最长重复子串：LCP 数组中最大值对应的两个后缀
function longestRepeatedSubstring(s) {
  const sa = buildSuffixArray(s);
  const lcp = buildLCP(s, sa);
  let maxLen = 0, maxIdx = 0;
  for (let i = 0; i < lcp.length; i++) {
    if (lcp[i] > maxLen) { maxLen = lcp[i]; maxIdx = i; }
  }
  return s.slice(sa[maxIdx], sa[maxIdx] + maxLen);
}

// 3. 不同子串计数：n*(n+1)/2 - sum(lcp)
function countDistinctSubstrings(s) {
  const sa = buildSuffixArray(s);
  const lcp = buildLCP(s, sa);
  const total = s.length * (s.length + 1) / 2;
  const lcpSum = lcp.reduce((a, b) => a + b, 0);
  return total - lcpSum;
}
```

## Kasai 算法详解

  LCP 数组的朴素构建是 O(n^2)，Kasai 算法利用了关键性质：`LCP[rank[i]] >= LCP[rank[i-1]] - 1`，将复杂度降到 O(n)。

## 复杂度总结

  | 操作 | 时间复杂度 | 说明 |
  | --- | --- | --- |
  | 构建后缀数组 | O(n log n) | 倍增法 |
  | 构建 LCP | O(n) | Kasai 算法 |
  | 子串查询 | O(m log n) | 二分查找 |
  | 最长重复子串 | O(n) | LCP 最大值 |
  | 不同子串计数 | O(n) | LCP 求和 |

  点击按钮查看结果
