# 06 - KMP 算法实现与优化 (KMP Implementation)

  ## 标准实现（含完整注释）

  以下是一个完整、健壮的 KMP 算法实现，包含边界处理：


```

/**
 * 计算模式串的前缀函数（π数组/next数组）
 * π[i] = P[0..i] 中最长公共前后缀的长度
 *
 * @param {string} P - 模式串
 * @returns {number[]} 前缀函数数组
 */
function computePrefixFunction(P) {
  const m = P.length;
  const pi = new Array(m).fill(0);

  // k 表示当前匹配的前缀长度
  // 同时也是最长公共前后缀的候选值
  let k = 0;

  // i 从 1 开始，因为 pi[0] 始终为 0
  for (let i = 1; i < m; i++) {
    // ★ 核心：当字符不匹配时，回退到更短的前缀
    // 利用已计算的 π 值，相当于在"前缀的前缀"中继续匹配
    while (k > 0 && P[i] !== P[k]) {
      k = pi[k - 1];
    }

    // 如果字符匹配，前缀长度加 1
    if (P[i] === P[k]) {
      k++;
    }

    // 记录当前 i 的前缀函数值
    pi[i] = k;
  }

  return pi;
}
  ```

  ## 优化：next 数组改进版

  经典 KMP 的 next 数组有一个优化：当 P[i] !== P[j] 回退后，如果 P[j] === P[next[j]]，可以继续回退，避免不必要的比较。


```

/**
 * 优化版 next 数组
 * 当回退后的字符与原字符相同时，继续回退
 */
function computeNextOptimized(P) {
  const m = P.length;
  const next = new Array(m).fill(0);
  next[0] = -1;   // 特殊标记：第一个字符失配时 i++, j++

  let i = 0, j = -1;
  while (i < m - 1) {
    if (j === -1 || P[i] === P[j]) {
      i++;
      j++;
      // ★ 优化：如果 P[i] === P[j]，则 next[i] = next[j]
      if (P[i] !== P[j]) {
        next[i] = j;
      } else {
        next[i] = next[j];
      }
    } else {
      j = next[j];
    }
  }

  return next;
}
  ```

  ## 多种语言实现

  ### JavaScript


```

function KMPSearch(text, pattern) {
  const n = text.length, m = pattern.length;
  if (m === 0) return [];
  if (m > n) return [];

  const pi = computePrefixFunction(pattern);
  const positions = [];
  let j = 0;

  for (let i = 0; i < n; i++) {
    while (j > 0 && text[i] !== pattern[j]) {
      j = pi[j - 1];
    }
    if (text[i] === pattern[j]) {
      j++;
    }
    if (j === m) {
      positions.push(i - m + 1);
      j = pi[j - 1];
    }
  }

  return positions;
}
  ```

  ### Python


```javascript

def compute_pi(pattern):
    m = len(pattern)
    pi = [0] * m
    k = 0
    for i in range(1, m):
        while k > 0 and pattern[i] != pattern[k]:
            k = pi[k - 1]
        if pattern[i] == pattern[k]:
            k += 1
        pi[i] = k
    return pi

def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0: return []
    pi = compute_pi(pattern)
    positions = []
    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = pi[j - 1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            positions.append(i - m + 1)
            j = pi[j - 1]
    return positions
  ```

  ## 边界情况处理









  | 场景 | 处理方式 |
| --- | --- |
| 空模式串 | 返回空数组 []（或根据需求返回 [0]） |
| 模式串比文本串长 | 返回空数组 [] |
| 模式串 = 文本串 | 返回 [0] |
| 重复字符模式（如 "AAAA"） | 前缀函数可正确处理，π = [0,1,2,3] |
| 无匹配 | 返回 [] |
| Unicode 字符 | charCodeAt() 可处理，但需要确保基数足够大 |

  ## 前缀函数可视化计算器
