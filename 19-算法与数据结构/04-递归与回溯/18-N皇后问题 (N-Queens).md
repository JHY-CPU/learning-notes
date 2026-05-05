## N-Queens


```javascript
在 N×N 棋盘上放置 N 个皇后，使其互不攻击。```


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


  点击按钮查看结果
