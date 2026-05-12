# 02 - 朴素字符串匹配 (Naive Matching)

## 算法思想

  朴素字符串匹配是最直观的匹配方法。它通过**滑动窗口**的方式，将模式串与文本串的每个子串逐一比较。


>
    **核心思路：** 从文本串的每一个可能位置开始，逐个字符与模式串比较。如果全部匹配则记录位置；否则移动一位，重新开始比较。


## 复杂度分析








  | 指标 | 值 |
| --- | --- |
| 最坏时间复杂度 | O(n·m) — 如 T="AAAAAA...A", P="AAA...AB" |
| 最好时间复杂度 | O(n) — 如 T="ABCDEFG", P="XYZ"（首位不匹配） |
| 平均时间复杂度 | O(n)（随机文本下） |
| 空间复杂度 | O(1) |
| 预处理 | 无 |


>
    **最坏情况举例：** T = "AAAAAAAAAAAAAAAAAB", P = "AAAAB"

    每次对齐都要比较到模式串的最后一个字符才发现不匹配，总比较次数约 (n-m+1)*m。


## 伪代码


```

NaiveStringMatching(T, P):
  n = T.length
  m = P.length
  for i = 0 to n - m:
    j = 0
    while j < m and T[i+j] == P[j]:
      j = j + 1
    if j == m:
      print "匹配位置:", i
  ```

## JavaScript 实现


```javascript

function naiveMatch(T, P) {
  const n = T.length, m = P.length;
  const positions = [];
  for (let i = 0; i <= n - m; i++) {
    let j = 0;
    while (j < m && T[i + j] === P[j]) j++;
    if (j === m) positions.push(i);
  }
  return positions;
}
  ```

## 与 KMP 的对比

```javascript
// 朴素匹配：O(n*m)
// KMP 预处理 O(m)，匹配 O(n)
// 当模式串较长或文本较大时，KMP 优势明显
```

## 改进思路

  ### 1. Rabin-Karp（哈希法）
  使用滚动哈希，O(n+m) 平均时间。将模式串哈希值与文本子串哈希值比较，哈希匹配时再逐字符验证。

```javascript
// Rabin-Karp 滚动哈希匹配
function rabinKarp(text, pattern) {
  const n = text.length, m = pattern.length;
  const base = 256, mod = 101;
  let patternHash = 0, textHash = 0, h = 1;

  // h = base^(m-1) % mod
  for (let i = 0; i < m - 1; i++) h = (h * base) % mod;

  for (let i = 0; i < m; i++) {
    patternHash = (base * patternHash + pattern.charCodeAt(i)) % mod;
    textHash = (base * textHash + text.charCodeAt(i)) % mod;
  }

  const positions = [];
  for (let i = 0; i <= n - m; i++) {
    if (patternHash === textHash) {
      // 哈希匹配，验证字符
      let match = true;
      for (let j = 0; j < m; j++) {
        if (text[i + j] !== pattern[j]) { match = false; break; }
      }
      if (match) positions.push(i);
    }
    // 滚动哈希
    if (i < n - m) {
      textHash = (base * (textHash - text.charCodeAt(i) * h) + text.charCodeAt(i + m)) % mod;
      if (textHash < 0) textHash += mod;
    }
  }
  return positions;
}
```

  ### 2. Boyer-Moore（跳跃法）
  从右向左匹配，利用坏字符和好后缀规则跳跃，实际效率最高。

## 算法选型指南

  | 算法 | 预处理 | 匹配 | 适用场景 |
  | --- | --- | --- | --- |
  | 朴素 | O(1) | O(nm) | 短模式、教学 |
  | KMP | O(m) | O(n) | 通用 |
  | Rabin-Karp | O(m) | O(n) 平均 | 多模式匹配 |
  | Boyer-Moore | O(m+σ) | O(n/m) 最好 | 实际文本搜索 |

## 交互演示

  输入文本和模式串，查看朴素匹配的完整过程：
