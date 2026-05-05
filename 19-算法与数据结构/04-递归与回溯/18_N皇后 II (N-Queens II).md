## N-Queens II


```javascript
N皇后问题的计数版本，只需返回不同解法的数量。```


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


  点击按钮查看结果
