# N-Queens


```javascript
在 N×N 棋盘上放置 N 个皇后，使其互不攻击。```

## 概念说明

经典回溯问题：在 N*N 棋盘上放置 N 个皇后，使得任意两个皇后不在同一行、同一列、同一条对角线上。皇后可以攻击同行、同列、同对角线上的任何棋子。N=8 时有 92 种解法。

## 核心思路

逐行放置皇后（保证不在同一行）。用三个集合（Set）标记已占用的列、主对角线（row-col 相同）、副对角线（row+col 相同）。每次放置前检查这三个约束，若冲突则跳过。放置后递归下一行，回溯时移除标记。这是约束满足问题（CSP）的经典示例。

## 复杂度分析

- **时间复杂度：** O(n!)，实际因剪枝远小于此。
- **空间复杂度：** O(n)，三个 Set + 递归栈。

## 适用场景

- 约束满足问题（CSP）教学
- 调度、排课、资源分配等约束求解
- 回溯算法的经典训练题

```
function solveNQueens(n) {
  const res = [];
  const board = new Array(n).fill(null).map(() => new Array(n).fill('.'));
  const cols = new Set(), diag1 = new Set(), diag2 = new Set();
  function backtrack(row) {
    if (row === n) { res.push(board.map(r => r.join(''))); return; }
    for (let col = 0; col < n; col++) {
      if (cols.has(col) || diag1.has(row-col) || diag2.has(row+col)) continue;
      board[row][col] = 'Q';
      cols.add(col); diag1.add(row-col); diag2.add(row+col);
      backtrack(row+1);
      board[row][col] = '.';
      cols.delete(col); diag1.delete(row-col); diag2.delete(row+col);
    }
  }
  backtrack(0);
  return res;
}
console.log(solveNQueens(4).length); // 2
console.log(solveNQueens(4)[0]); // [".Q..","...Q","Q...","..Q."]```


## 常见变体与技巧

- **位运算加速：** 用整数 bit 代替 Set，`col | diag1 | diag2` 做位或判断冲突。
- **对称性剪枝：** 第一行只需遍历一半列（利用棋盘对称性），结果乘以 2。
- **数独问题：** 与 N 皇后类似的 CSP 问题，用回溯 + 约束传播求解。

  点击按钮查看结果
