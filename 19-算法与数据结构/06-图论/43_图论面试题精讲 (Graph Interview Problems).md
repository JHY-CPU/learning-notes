## Graph Interview Problems


```javascript
图论面试高频题分类与解题思路。```


```
// 图论面试高频题型
// 1. 图的遍历：克隆图、岛屿数量
// 2. 拓扑排序：课程表、任务调度
// 3. 最短路径：网络延迟时间、单词接龙
// 4. 连通性：省份数量、冗余连接
// 5. 环检测：无向图/有向图检测
// 6. 二分图：可能的二分法
// 7. 最小生成树：连接所有城市的最小费用
// 8. 并查集：等式方程的可满足性

// 解题模板：BFS 层序遍历
function numIslands(grid) {
  let count = 0;
  for (let i = 0; i < grid.length; i++)
    for (let j = 0; j < grid[0].length; j++)
      if (grid[i][j] === '1') { count++; dfs(grid, i, j); }
  return count;
}
function dfs(grid, i, j) {
  if (i<0||i>=grid.length||j<0||j>=grid[0].length||grid[i][j]!=='1') return;
  grid[i][j] = '0';
  dfs(grid,i+1,j); dfs(grid,i-1,j); dfs(grid,i,j+1); dfs(grid,i,j-1);
}
console.log(numIslands([
  ['1','1','0','0','0'],
  ['1','1','0','0','0'],
  ['0','0','1','0','0'],
  ['0','0','0','1','1']
])); // 3```


  点击按钮查看结果
