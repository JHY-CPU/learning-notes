# Word Search


```javascript
在二维网格中搜索单词是否存在，回溯探索相邻单元格。```

## 概念说明

给定 m*n 的字符网格和一个单词，判断单词是否存在于网格中。单词可以由相邻单元格（上下左右）的字母构成，同一位置的单元格不能重复使用。这是回溯在二维图上的应用。

## 核心思路

遍历网格每个位置作为起点。DFS 递归时用 `#` 标记已访问单元格（原地修改，无需额外 visited 数组），递归四个方向查找下一个字符。回溯时恢复原字符。若当前字符不匹配则直接剪枝。对长单词可先预处理单词中各字符出现频率进行预筛选。

## 复杂度分析

- **时间复杂度：** O(m * n * 3^L)，L 为单词长度，每个位置最多 3 个方向（不含来源方向）。
- **空间复杂度：** O(L)，递归栈深度为单词长度。

## 适用场景

- 单词搜索游戏（Boggle）
- 矩阵中的路径搜索
- 字典前缀树（Trie）+ 回溯的结合题型

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


## 常见变体与技巧

- **单词搜索 II（多单词）：** 用 Trie 存储所有单词，DFS 遍历时同时匹配前缀，一次遍历找所有单词。
- **visited 数组 vs 原地标记：** 原地标记省空间，但注意回溯时必须恢复原值。
- **优化方向：** 对长单词优先从出现频率低的字符开始搜索，减少搜索范围。

  点击按钮查看结果
