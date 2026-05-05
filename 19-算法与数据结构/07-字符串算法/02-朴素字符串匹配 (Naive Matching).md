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

  ## 交互演示

  输入文本和模式串，查看朴素匹配的完整过程：
