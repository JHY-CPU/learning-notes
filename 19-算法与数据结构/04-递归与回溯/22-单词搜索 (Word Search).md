## Word Search


```javascript
在二维网格中搜索单词是否存在，回溯探索相邻单元格。```


```
function exist(board, word) {
  const m = board.length, n = board[0].length;
  function dfs(i, j, idx) {
    if (idx === word.length) return true;
    if (i<0||i>=m||j<0||j>=n||board[i][j]!==word[idx]) return false;
    const tmp = board[i][j];
    board[i][j] = '#';
    const found = dfs(i+1,j,idx+1)||dfs(i-1,j,idx+1)||dfs(i,j+1,idx+1)||dfs(i,j-1,idx+1);
    board[i][j] = tmp;
    return found;
  }
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      if (dfs(i, j, 0)) return true;
  return false;
}
const board = [
  ['A','B','C','E'],
  ['S','F','C','S'],
  ['A','D','E','E']
];
console.log(exist(board, 'ABCCED')); // true
console.log(exist(board, 'SEE')); // true
console.log(exist(board, 'ABCB')); // false```


  点击按钮查看结果
