## Sudoku Solver


```javascript
数独求解是回溯的经典应用，需满足行、列、宫格的数字不重复约束。```


```
function solveSudoku(board) {
  function isValid(board, row, col, num) {
    for (let i = 0; i < 9; i++) {
      if (board[row][i] === num) return false;
      if (board[i][col] === num) return false;
      const r = Math.floor(row/3)*3 + Math.floor(i/3);
      const c = Math.floor(col/3)*3 + i%3;
      if (board[r][c] === num) return false;
    }
    return true;
  }
  function backtrack() {
    for (let i = 0; i < 9; i++)
      for (let j = 0; j < 9; j++)
        if (board[i][j] === '.') {
          for (let num = 1; num <= 9; num++) {
            const ch = String(num);
            if (isValid(board, i, j, ch)) {
              board[i][j] = ch;
              if (backtrack()) return true;
              board[i][j] = '.';
            }
          }
          return false;
        }
    return true;
  }
  backtrack();
  return board;
}
const board = [
  ['5','3','.','.','7','.','.','.','.'],
  ['6','.','.','1','9','5','.','.','.'],
  ['.','9','8','.','.','.','.','6','.'],
  ['8','.','.','.','6','.','.','.','3'],
  ['4','.','.','8','.','3','.','.','1'],
  ['7','.','.','.','2','.','.','.','6'],
  ['.','6','.','.','.','.','2','8','.'],
  ['.','.','.','4','1','9','.','.','5'],
  ['.','.','.','.','8','.','.','7','9']
];
console.log(solveSudoku(board));```


  点击按钮查看结果
