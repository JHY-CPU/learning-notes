# N-Queens II


```javascript
N皇后问题的计数版本，只需返回不同解法的数量。```

## 概念说明

与 N 皇后问题相同，但不需要输出具体棋盘布局，只需返回解法总数。由于省去了棋盘存储和字符串拼接，实际运行速度更快。

## 核心思路

与 N 皇后相同的回溯框架，但将 `res.push(...)` 简化为 `count++`。使用位运算可进一步优化：用整数代替 Set 表示列和对角线的占用状态，通过位操作判断冲突，大幅提升常数因子。

## 复杂度分析

- **时间复杂度：** O(n!)，与 N 皇后相同。
- **空间复杂度：** O(n)，递归栈 + 三个标记集合。位运算优化后为 O(1) 额外空间。

## 适用场景

- 需要统计方案数而非输出具体方案的场景
- 位运算在回溯中的应用示范
- 性能对比：体会减少存储开销带来的速度提升

```
function totalNQueens(n) {
  let count = 0;
  const cols = new Set(), diag1 = new Set(), diag2 = new Set();
  function backtrack(row) {
    if (row === n) { count++; return; }
    for (let col = 0; col < n; col++) {
      if (cols.has(col) || diag1.has(row-col) || diag2.has(row+col)) continue;
      cols.add(col); diag1.add(row-col); diag2.add(row+col);
      backtrack(row+1);
      cols.delete(col); diag1.delete(row-col); diag2.delete(row+col);
    }
  }
  backtrack(0);
  return count;
}
console.log(`4皇后: ${totalNQueens(4)}`); // 2
console.log(`8皇后: ${totalNQueens(8)}`); // 92```


## 常见变体与技巧

- **位运算核心：** `available = (~(col | ld | rd)) & mask`，用 lowbit 提取最低位的可放位置。
- **已知 N 皇后的解数：** N=1~15 的解数为 1,0,0,2,10,4,40,92,352,724,2680,14200,73712,365597,2279184。
- **面试技巧：** 先写 N 皇后（输出方案），再优化为 N 皇后 II（只计数），展示优化思路。

## 位运算优化实现

```javascript
// 位运算优化的 N 皇后 II
function totalNQueensBitwise(n) {
  let count = 0;
  const mask = (1 << n) - 1;  // n 位全 1

  function solve(col, diag1, diag2) {
    if (col === mask) { count++; return; }
    // 可放位置：不在任何攻击范围内的位
    let available = ~(col | diag1 | diag2) & mask;
    while (available) {
      // 取最低位的 1
      const pos = available & (-available);
      available ^= pos;  // 移除该位
      solve(
        col | pos,
        (diag1 | pos) << 1,
        (diag2 | pos) >> 1
      );
    }
  }

  solve(0, 0, 0);
  return count;
}
```

## N 皇后解数（已知值）

  | N | 解数 |
  | --- | --- |
  | 1 | 1 |
  | 2 | 0 |
  | 3 | 0 |
  | 4 | 2 |
  | 5 | 10 |
  | 6 | 4 |
  | 7 | 40 |
  | 8 | 92 |
  | 9 | 352 |
  | 10 | 724 |
  | 11 | 2680 |
  | 12 | 14200 |

## N 皇后 I 对比

  N 皇后 I 需要输出具体棋盘布局，需要用二维数组记录每行皇后的列号。N 皇后 II 只需计数，省去了棋盘存储和方案收集，速度更快。

  点击按钮查看结果
