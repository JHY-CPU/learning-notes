# 26-栈与深度优先搜索 (Stack & DFS)

栈可以模拟递归实现深度优先搜索（DFS），避免递归调用栈溢出，且更灵活地控制搜索过程。

## 用栈实现 DFS

```javascript
// 图的 DFS（显式栈版）
function dfs(graph, start) {
  const visited = new Set();
  const stack = [start];
  const result = [];

  while (stack.length) {
    const node = stack.pop();
    if (!visited.has(node)) {
      visited.add(node);
      result.push(node);
      // 逆序压栈保证正序访问
      const neighbors = graph[node] || [];
      for (let i = neighbors.length - 1; i >= 0; i--) {
        if (!visited.has(neighbors[i])) stack.push(neighbors[i]);
      }
    }
  }
  return result;
}

const g = { A: ['B', 'C'], B: ['D'], C: ['E'], D: [], E: [] };
console.log(dfs(g, 'A')); // ['A', 'B', 'D', 'C', 'E']
```

## C++ 实现

```cpp
#include <vector>
#include <stack>
#include <unordered_set>
using namespace std;

void dfs(vector<vector<int>>& graph, int start) {
    unordered_set<int> visited;
    stack<int> stk;
    stk.push(start);

    while (!stk.empty()) {
        int node = stk.top(); stk.pop();
        if (visited.count(node)) continue;
        visited.insert(node);
        printf("访问节点: %d\n", node);

        for (int i = graph[node].size() - 1; i >= 0; i--) {
            if (!visited.count(graph[node][i])) {
                stk.push(graph[node][i]);
            }
        }
    }
}
```

## 二维网格 DFS

```javascript
// 岛屿数量问题
function numIslands(grid) {
  let count = 0;
  const rows = grid.length, cols = grid[0].length;

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      if (grid[i][j] === '1') {
        count++;
        // 用栈实现 DFS 置零
        const stack = [[i, j]];
        while (stack.length) {
          const [r, c] = stack.pop();
          if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] !== '1') continue;
          grid[r][c] = '0';
          stack.push([r+1, c], [r-1, c], [r, c+1], [r, c-1]);
        }
      }
    }
  }
  return count;
}
```

## DFS vs BFS 与栈/队列

```
DFS：用栈（递归栈或显式栈），适合搜索所有路径、回溯
BFS：用队列，适合最短路径、层次遍历
```

| 特性 | DFS (栈) | BFS (队列) |
|------|---------|-----------|
| 数据结构 | 栈 | 队列 |
| 空间 | O(h) 树高 | O(w) 最大宽度 |
| 最短路径 | 不保证 | 保证（无权图） |
| 适用 | 路径搜索、拓扑排序 | 层次遍历、最短路径 |

## 迭代加深 DFS

```javascript
// 迭代加深搜索：限制深度的 DFS
function iddfs(graph, start, maxDepth) {
  for (let depth = 0; depth <= maxDepth; depth++) {
    const visited = new Set();
    const result = dls(graph, start, depth, visited);
    if (result.length) return result;
  }
  return [];
}

function dls(graph, node, depth, visited) {
  if (depth === 0) return [node];
  visited.add(node);
  let result = [];
  for (const nei of graph[node] || []) {
    if (!visited.has(nei)) {
      result = result.concat(dls(graph, nei, depth - 1, visited));
    }
  }
  return result;
}
```

## 常见应用

- 图/树的遍历与搜索
- 拓扑排序
- 连通分量检测
- 岛屿/区域计数
- 路径查找与回溯
- 迷宫求解

## 常见陷阱

1. **无限循环**：忘记标记已访问节点
2. **逆序压栈**：DFS 用栈时邻居的压栈顺序影响访问顺序
3. **栈溢出**：深度很大的图用递归 DFS 可能溢出，改用显式栈
4. **visited 重置**：某些问题需要在不同起点重置 visited
