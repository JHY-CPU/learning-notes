# Graph Modeling

  将实际问题转化为图论问题，是图论解题的关键步骤。

## 什么是图论建模

  图论建模是将现实世界的问题抽象为图的顶点和边的过程。建模的好坏直接决定了能否用图论算法解决问题。关键在于识别问题中的"实体"（顶点）和"关系"（边）。

## 常见建模模式

    - **网格 -> 图：**每个格子是顶点，相邻格子连边（四方向或八方向）
    - **依赖关系 -> 有向图：**拓扑排序求解执行顺序
    - **等价关系 -> 无向图：**并查集或连通分量
    - **状态空间 -> 隐式图：**BFS/DFS 搜索所有可达状态
    - **约束满足 -> 二分图：**最大匹配或匈牙利算法


```
// 常见图论建模技巧
//
// 1. 网格 → 图
//    每个格子是顶点，四方向/八方向连边
//
// 2. 依赖关系 → 有向图
//    拓扑排序求解
//
// 3. 等价关系 → 无向图
//    并查集/连通分量
//
// 4. 状态空间 → 隐式图
//    BFS/DFS 搜索所有状态
//
// 5. 约束满足 → 二分图
//    最大匹配/匈牙利算法
console.log('图论建模是将问题转化为顶点和边的艺术');
```


## 复杂度分析

    - 建模本身通常 O(问题规模)
    - 算法复杂度取决于选择的图论算法

## 常见陷阱

    - 顶点和边的定义不准确导致问题无法正确转化
    - 忽略图的方向性（有向 vs 无向）
    - 状态空间爆炸：隐式图的状态数可能指数级增长

## 建模实例详解

```javascript
// 例 1：网格最短路
// LeetCode 1091. 二进制矩阵中的最短路径
// 每个格子是顶点，八方向连边，BFS 求最短路
function shortestPathBinaryMatrix(grid) {
  const n = grid.length;
  if (grid[0][0] || grid[n-1][n-1]) return -1;
  const dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
  const queue = [[0, 0, 1]];  // [row, col, steps]
  grid[0][0] = 1;  // 标记已访问
  while (queue.length) {
    const [r, c, steps] = queue.shift();
    if (r === n - 1 && c === n - 1) return steps;
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < n && nc >= 0 && nc < n && grid[nr][nc] === 0) {
        grid[nr][nc] = 1;
        queue.push([nr, nc, steps + 1]);
      }
    }
  }
  return -1;
}

// 例 2：状态空间搜索
// LeetCode 752. 打开转盘锁
// 每个状态是一个顶点，每次拨一位得到邻居顶点
function openLock(deadends, target) {
  const dead = new Set(deadends);
  if (dead.has('0000')) return -1;
  if (target === '0000') return 0;

  const visited = new Set(['0000']);
  const queue = [['0000', 0]];

  while (queue.length) {
    const [state, steps] = queue.shift();
    for (let i = 0; i < 4; i++) {
      for (const d of [-1, 1]) {
        const arr = state.split('');
        arr[i] = String((+arr[i] + d + 10) % 10);
        const next = arr.join('');
        if (next === target) return steps + 1;
        if (!visited.has(next) && !dead.has(next)) {
          visited.add(next);
          queue.push([next, steps + 1]);
        }
      }
    }
  }
  return -1;
}
```

## 建模检查清单

  1. **顶点是什么？**实体、状态、还是位置？
  2. **边是什么？**关系、转移、还是操作？
  3. **有向还是无向？**关系是否对称？
  4. **是否加权？**权重的含义是什么？
  5. **需要什么算法？**最短路、连通性、还是流？
  6. **规模如何？**决定算法选择

## 常见建模模式汇总

  | 问题类型 | 顶点 | 边 | 算法 |
  | --- | --- | --- | --- |
  | 迷宫最短路 | 格子 | 相邻格子 | BFS |
  | 课程安排 | 课程 | 先修关系 | 拓扑排序 |
  | 最少换乘 | 站点 | 同一线路连边 | BFS |
  | 任务分配 | 工人+任务 | 可分配关系 | 匈牙利 |
  | 最大流量 | 原图顶点 | 原图边（有容量） | Dinic |
  | 拼图游戏 | 状态 | 一步操作 | BFS/DFS |

## 实际应用

  - **社交网络：**用户是顶点，好友关系是边，社区发现和影响力分析
  - **电路设计：**元件是顶点，导线是边，最短路径和流算法优化信号传输
  - **交通规划：**路口是顶点，道路是边，最短路和流量优化
  - **推荐系统：**用户和商品是二部图，最大匹配和随机游走做推荐

## LeetCode 建模专题

  - 743. 网络延迟时间（加权有向图 + Dijkstra）
  - 994. 腐烂的橘子（多源 BFS）
  - 127. 单词接龙（隐式图 BFS）
  - 207. 课程表（有向图拓扑排序）
