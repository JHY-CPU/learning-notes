# 05 - KMP 算法 — 匹配过程 (KMP: Matching)

  ## 从前缀函数到匹配

  KMP 匹配过程利用前缀函数，在字符失配时将模式串**回退到适当的位置**，而不是从头开始。这样，文本串的指针 **永远不会回退**。


>
    **匹配阶段的核心思想：**

      - 维护指针 i（文本串）和 j（模式串）

      - 当 T[i] === P[j] 时，i++，j++

      - 当 T[i] !== P[j] 且 j > 0 时，j = π[j-1]（利用前缀函数回退）

      - 当 T[i] !== P[j] 且 j === 0 时，i++（模式串第一个字符就不匹配）

      - 当 j === m 时，找到一个匹配，记录位置 i-m，然后 j = π[j-1] 继续查找下一个匹配




  ## 完整 KMP 算法


```

function KMPSearch(text, pattern) {
  const n = text.length, m = pattern.length;
  if (m === 0) return [];
  const pi = computePrefixFunction(pattern);
  const positions = [];
  let j = 0;  // 模式串指针

  for (let i = 0; i < n; i++) {
    // 字符失配时，通过 π 数组回退 j
    while (j > 0 && text[i] !== pattern[j]) {
      j = pi[j - 1];
    }
    if (text[i] === pattern[j]) {
      j++;
    }
    if (j === m) {
      positions.push(i - m + 1);
      j = pi[j - 1];  // 继续查找
    }
  }
  return positions;
}

function computePrefixFunction(P) {
  const m = P.length;
  const pi = new Array(m).fill(0);
  let k = 0;
  for (let i = 1; i < m; i++) {
    while (k > 0 && P[i] !== P[k]) k = pi[k - 1];
    if (P[i] === P[k]) k++;
    pi[i] = k;
  }
  return pi;
}
  ```

  ## 匹配示例

  T = "ABABDABACDABABCABAB", P = "ABABCABAB"

  执行过程直观展示：


```

i=0  A B A B D A B A C D A B A B C A B A B     j=0
     ↑
     A B A B C A B A B
     ↑
     j=0 → 匹配 → j=4 时失配 (T[4]=D ≠ P[4]=C)
     j = π[3] = 2

i=4  A B A B D A B A C D A B A B C A B A B     j=2
         ↑
       A B A B C A B A B
         ↑
       P[2]=A ≠ T[4]=D → j = π[1] = 0

i=4  A B A B D A B A C D A B A B C A B A B     j=0
           ↑
           A B A B C A B A B
           ↑
           P[0]=A ≠ T[4]=D → i++

i=5  A B A B D A B A C D A B A B C A B A B     j=0
             ↑
             A B A B C A B A B
             ↑
             P[0]=A === T[5]=A → j=1 → ...
  ```

  ## 复杂度分析





    ****

  | 指标 | 值 |
| --- | --- |
| 预处理时间（前缀函数） | O(m) |
| 匹配时间 | O(n) |
| 总时间复杂度 | O(n + m) — |
| 空间复杂度 | O(m) — 存储 π 数组 |


>
    **为什么是 O(n)？**

    虽然 while 循环可能多次回退 j，但 j 在整个过程中最多增加 n 次，而每次回退（j = π[j-1]）都会减少 j。由于 j ≥ 0，回退的总次数不会超过 j 的增加次数，因此总复杂度为 O(n)。


  ## 交互演示：KMP 完整匹配
