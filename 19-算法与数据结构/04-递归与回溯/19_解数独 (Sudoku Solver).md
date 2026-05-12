# 20-解数独 (Sudoku Solver)

数独求解是回溯的经典应用，需满足行、列、宫格的数字不重复约束。

## 概念说明

数独是一个 9x9 网格，需要在空格中填入 1-9 的数字，使得每行、每列、每个 3x3 宫格内的数字都不重复。这是一个约束满足问题（CSP），用回溯搜索逐个尝试填入数字。

## 复杂度分析

| 指标 | 值 |
|------|-----|
| 最坏时间 | O(9^(空格数)) |
| 实际 | 远小于此（剪枝效果显著） |
| 空间 | O(81) 递归栈 |

## JavaScript 实现

```javascript
function solveSudoku(board) {
  // 检查在 (row, col) 放置 num 是否合法
  function isValid(row, col, num) {
    for (let i = 0; i < 9; i++) {
      if (board[row][i] === num) return false;       // 行不重复
      if (board[i][col] === num) return false;       // 列不重复
      // 宫格不重复
      const r = Math.floor(row / 3) * 3 + Math.floor(i / 3);
      const c = Math.floor(col / 3) * 3 + (i % 3);
      if (board[r][c] === num) return false;
    }
    return true;
  }

  function backtrack() {
    for (let i = 0; i < 9; i++) {
      for (let j = 0; j < 9; j++) {
        if (board[i][j] === '.') {
          for (let num = 1; num <= 9; num++) {
            const ch = String(num);
            if (isValid(i, j, ch)) {
              board[i][j] = ch;
              if (backtrack()) return true;
              board[i][j] = '.';  // 回溯
            }
          }
          return false; // 1-9 都不行，无解
        }
      }
    }
    return true; // 所有格子填完
  }

  backtrack();
  return board;
}

// 测试
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
console.log(solveSudoku(board));
```

## C++ 实现

```cpp
#include <vector>
#include <string>
using namespace std;

class SudokuSolver {
public:
    bool solveSudoku(vector<vector<char>>& board) {
        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] == '.') {
                    for (char c = '1'; c <= '9'; c++) {
                        if (isValid(board, i, j, c)) {
                            board[i][j] = c;
                            if (solveSudoku(board)) return true;
                            board[i][j] = '.';
                        }
                    }
                    return false;
                }
            }
        }
        return true;
    }

    bool isValid(vector<vector<char>>& board, int row, int col, char c) {
        for (int i = 0; i < 9; i++) {
            if (board[row][i] == c) return false;
            if (board[i][col] == c) return false;
            int r = 3 * (row / 3) + i / 3;
            int cc = 3 * (col / 3) + i % 3;
            if (board[r][cc] == c) return false;
        }
        return true;
    }
};
```

## 优化技巧

1. **候选数预计算**：预计算每个空格的候选数字，减少 isValid 调用
2. **选择候选数最少的格子**：优先填入约束最多的格子（MRV 启发式）
3. **约束传播**：当某格填入数字后，立即更新同行/列/宫的候选数
4. **位运算**：用 bitset 表示候选数，加速交集运算

## 常见陷阱

1. **宫格索引计算**：`3*(row/3) + i/3` 容易写错
2. **回溯恢复**：必须在回溯时恢复为 `.`，否则影响其他分支
3. **修改原数组**：解数独会原地修改 board

## 实际应用

数独求解与调度问题、资源分配问题本质相同，都是 CSP。掌握回溯解数独有助于理解更复杂的约束求解场景。
