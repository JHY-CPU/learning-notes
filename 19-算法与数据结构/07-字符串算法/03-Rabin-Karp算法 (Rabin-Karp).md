# 03 - Rabin-Karp 算法 (Rabin-Karp)

  ## 算法思想

  Rabin-Karp 算法利用**哈希函数**将字符串映射为数值，通过比较哈希值来快速判断子串是否可能匹配。


>
    **核心思路：**

      - 计算模式串的哈希值 hash(P)

      - 计算文本串中每个长度为 m 的子串的哈希值

      - 如果哈希值相等，再逐个字符验证（解决哈希冲突）

      - 利用**滚动哈希**（Rolling Hash）在 O(1) 时间内从上一个子串的哈希推算出下一个子串的哈希




  ## 滚动哈希公式

  使用基数进制（base = d，通常取 256 或一个大质数），模数 q（一个大质数，如 109+7）：


```javascript

// 计算哈希值
hash(s) = (s[0]*d^(m-1) + s[1]*d^(m-2) + ... + s[m-1]) mod q

// 滚动更新（窗口右移一位）
new_hash = ((old_hash - T[i]*d^(m-1)) * d + T[i+m]) mod q
// 如果 new_hash < 0，需要加 q 调整
  ```

  ## 算法步骤


    - **预处理：**计算模式串哈希 hashP，计算 dm-1 mod q

    - **计算第一个窗口：**计算 T[0..m-1] 的哈希 hashT

    - **滑动比较：**

        - 如果 hashT === hashP，逐个字符比较确认是否真匹配

        - 用滚动哈希更新 hashT 得到下一个窗口的哈希




    - 重复直到文本串末尾



  ## 复杂度分析







  | 指标 | 值 |
| --- | --- |
| 预处理时间 | O(m) |
| 最坏匹配时间 | O(n·m)（大量哈希冲突时） |
| 平均匹配时间 | O(n+m)（良好哈希函数下） |
| 空间复杂度 | O(1) |

  ## JavaScript 实现


```

function rabinKarp(text, pattern) {
  const d = 256;   // 基数（字符集大小）
  const q = 101;   // 模数（大质数）
  const n = text.length, m = pattern.length;
  const positions = [];
  if (m > n) return positions;

  let hashP = 0, hashT = 0, h = 1;

  // 计算 d^(m-1) mod q
  for (let i = 0; i < m - 1; i++) h = (h * d) % q;

  // 计算模式串和第一个窗口的哈希
  for (let i = 0; i < m; i++) {
    hashP = (hashP * d + pattern.charCodeAt(i)) % q;
    hashT = (hashT * d + text.charCodeAt(i)) % q;
  }

  // 滑动匹配
  for (let i = 0; i <= n - m; i++) {
    if (hashP === hashT) { // 哈希相同，逐个字符验证
      let j = 0;
      while (j < m && text[i + j] === pattern[j]) j++;
      if (j === m) positions.push(i);
    }
    // 计算下一个窗口的哈希
    if (i < n - m) {
      hashT = ((hashT - text.charCodeAt(i) * h) * d + text.charCodeAt(i + m)) % q;
      if (hashT < 0) hashT += q; // 处理负数
    }
  }
  return positions;
}
  ```

  ## 交互演示
