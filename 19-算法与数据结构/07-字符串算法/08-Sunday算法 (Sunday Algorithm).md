# 08 - Sunday 算法 (Sunday Algorithm)

  ## 算法思想

  Sunday 算法是 Boyer-Moore 算法的简化变体，由 Daniel M. Sunday 在 1990 年提出。它同样**从右向左**比较字符，但在决定跳跃距离时只关注**下一个字符**（即文本串中当前窗口右侧紧邻的字符）。


>
    **关键思想：**


      - 从左到右移动模式串，从右到左比较字符

      - 如果当前对齐完全匹配，记录位置

      - 否则，查看文本串中**模式串右侧下一个字符** T[i+m]

      - 如果 T[i+m] 在模式串中出现，将模式串右移使该字符与模式串中**最右**的相同字符对齐

      - 如果 T[i+m] 不在模式串中，将模式串右移 m+1 个字符




  ## 与 Boyer-Moore 的区别









  | 特性 | Boyer-Moore | Sunday |
| --- | --- | --- |
| 决定跳跃的字符 | 失配字符（坏字符）+ 好后缀 | 窗口右侧下一个字符 |
| 规则数量 | 坏字符 + 好后缀（两种规则） | 一种简单规则 |
| 实现复杂度 | 复杂（特别是好后缀表） | 非常简单 |
| 最坏情况 | O(n·m) | O(n·m) |
| 平均性能 | 非常好 | 接近 BM，有时更好 |
| 字符集 | 需要构建坏字符表 | 需要构建偏移表 |

  ## 算法实现


```

/**
 * Sunday 字符串匹配算法
 *
 * @param {string} text - 文本串
 * @param {string} pattern - 模式串
 * @returns {number[]} 匹配位置数组
 */
function sundayMatch(text, pattern) {
  const n = text.length, m = pattern.length;
  if (m === 0 || m > n) return [];
  const positions = [];

  // 预处理：构建偏移表（shift table）
  // shift[c] = 字符c在模式串中从右数第一次出现离末尾的距离
  // 若字符不在模式串中，则 shift = m + 1
  const shift = {};
  for (let i = 0; i < m; i++) {
    // 偏移量 = 从右到左的距离 = m - i
    shift[pattern[i]] = m - i;
  }

  let i = 0;  // 文本串指针
  while (i <= n - m) {
    // 从右向左比较
    let j = 0;
    while (j < m && pattern[m - 1 - j] === text[i + m - 1 - j]) {
      j++;
    }

    if (j === m) {
      // 完全匹配
      positions.push(i);
    }

    // 根据下一个字符决定跳跃距离
    if (i + m < n) {
      const nextChar = text[i + m];
      i += shift[nextChar] !== undefined ? shift[nextChar] : (m + 1);
    } else {
      break;
    }
  }

  return positions;
}
  ```

  ## 偏移表计算示例

  模式串 P = **"SEARCH"**


```

字符 | 最右位置(从0起) | shift = m - pos
'S'  | 0              | 6 - 0 = 6
'E'  | 1              | 6 - 1 = 5
'A'  | 2              | 6 - 2 = 4
'R'  | 3              | 6 - 3 = 3
'C'  | 4              | 6 - 4 = 2
'H'  | 5              | 6 - 5 = 1
其他 | —              | m + 1 = 7
  ```


>
    当字符在模式串中出现多次时，最右的（最靠近末尾的）位置决定 shift，因为 shift[char] 会被后来的覆盖。


  ## 复杂度分析







  | 指标 | 值 |
| --- | --- |
| 预处理时间 | O(m) |
| 最坏匹配时间 | O(n·m) |
| 平均匹配时间 | O(n/m) — 对自然语言文本非常高效 |
| 空间复杂度 | O(Σ) — 字符集大小 |

  ## 交互演示
