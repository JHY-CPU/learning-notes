# 07 - Boyer-Moore 算法 (Boyer-Moore)

  ## 算法思想

  Boyer-Moore（BM）算法是实践中最高效的字符串匹配算法之一。它的核心思想是：**从右向左**扫描模式串，利用两种启发式规则跳过尽可能多的字符。


>
    **核心特征：**

      - **从右向左**比较字符（与 KMP 和朴素算法相反）

      - 利用**坏字符规则**（Bad Character Rule）

      - 利用**好后缀规则**（Good Suffix Rule）

      - 取两种规则中跳跃距离**较大**者进行移动




  ## 坏字符规则（Bad Character Rule）

  当 T[i+j]（文本字符）与 P[j]（模式字符）不匹配时：


    - 在模式串 P 中查找 T[i+j] 的**最右出现位置**（排除当前位置 j）

    - 如果找到，将模式串右移使该字符对齐

    - 如果没找到，将模式串右移到文本字符之后



```javascript

// 预处理：构建坏字符表
// bc[c] = 字符 c 在模式串中最右出现的位置
function buildBadCharTable(P) {
  const bc = {};
  for (let i = 0; i < P.length; i++) {
    bc[P[i]] = i;  // 相同字符后出现的覆盖前面的
  }
  return bc;
}
  ```

  ## 好后缀规则（Good Suffix Rule）

  当模式串的后缀 P[j+1..m-1] 已经匹配，但 P[j] 与 T[i+j] 不匹配时：


    - 在模式串中查找**与已匹配后缀相同**的子串，且前一个字符不同

    - 如果找到，将模式串右移到该子串对齐已匹配后缀的位置

    - 如果找不到，查找已匹配后缀的**最长前缀**也是模式串的后缀



  ## 完整实现


```

function boyerMoore(text, pattern) {
  const n = text.length, m = pattern.length;
  if (m === 0) return [];
  const positions = [];

  // 预处理坏字符表
  const bc = {};
  for (let i = 0; i < m; i++) bc[pattern[i]] = i;

  // 预处理好后缀表
  const suffix = new Array(m).fill(0);
  const prefix = new Array(m).fill(false);
  for (let i = 0; i < m - 1; i++) {
    let j = i;
    let k = 0;
    while (j >= 0 && pattern[j] === pattern[m - 1 - k]) {
      j--;
      k++;
      suffix[k] = j + 1;
    }
    if (j < 0) prefix[k] = true;
  }

  let i = 0; // 文本串指针
  while (i <= n - m) {
    let j = m - 1; // 模式串指针（从右向左）
    while (j >= 0 && pattern[j] === text[i + j]) j--;

    if (j < 0) {
      // 完全匹配
      positions.push(i);
      // 移动模式串
      const gs = m - (m > 1 ? suffix[m - 1] + 1 : 0);
      i += gs;
    } else {
      // 坏字符规则
      const badChar = bc[text[i + j]] !== undefined ? bc[text[i + j]] : -1;
      const bcShift = j - badChar;
      if (bcShift < 0) bcShift = 1;

      // 好后缀规则
      let gsShift = 0;
      const k = m - 1 - j; // 已匹配的后缀长度
      if (k > 0) {
        if (suffix[k] !== 0) {
          gsShift = j - suffix[k] + 1;
        } else {
          // 查找最长前缀匹配
          for (let r = k - 1; r > 0; r--) {
            if (prefix[r]) { gsShift = m - r; break; }
          }
          if (gsShift === 0) gsShift = m;
        }
      } else {
        gsShift = 1;
      }

      i += Math.max(bcShift, gsShift);
    }
  }
  return positions;
}
  ```

  ## 复杂度分析








  | 指标 | 值 |
| --- | --- |
| 预处理时间 | O(m + Σ) — Σ 是字符集大小 |
| 最坏匹配时间 | O(n·m)（不加好后缀规则时） |
| 最佳匹配时间 | O(n/m) — 每次跳跃 m 个字符 |
| 平均匹配时间 | O(n/m) + O(m) — 对自然语言非常高效 |
| 空间复杂度 | O(m + Σ) |


>
    **实践中的性能：** 对于英文章节文本，BM 算法通常只需要检查约 n/m 个字符，远快于 KMP。这也是为什么许多文本编辑器的"查找"功能使用 BM 算法或其变体。


  ## 交互演示

  BM 算法匹配演示（带坏字符和好后缀跟踪）：
