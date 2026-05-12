# 图遍历对比 (BFS vs DFS)

  BFS 和 DFS 是图遍历的两种基本策略，各有优劣。理解它们的区别是选择正确算法的前提。

## 核心区别

  BFS 使用队列按层遍历，DFS 使用栈（或递归）按深度遍历。BFS 保证先访问近的顶点，DFS 保证沿一条路径深入到底再回溯。





  | 特性 | BFS | DFS |
| --- | --- | --- |
| 数据结构 | 队列 (Queue) | 栈 (Stack) / 递归 |
| 空间复杂度 | O(V) 最坏 | O(V) 最坏 |
| 最短路径（无权图） | 是 | 否 |
| 连通分量 | 可以 | 可以 |
| 拓扑排序 | Kahn 算法 | DFS 后序 |
| 环检测 | 可以 | 可以 |
| 内存使用 | 存储整层 | 存储一条路径 |

## 选择指南


    - 需要最短路径 **→** BFS

    - 需要探索所有路径 **→** DFS

    - 树/图为分层结构 **→** BFS

    - 检测环路 **→** DFS

    - 内存受限但很深 **→** BFS

## 复杂度对比

  两者时间复杂度都是 O(V+E)。空间方面，BFS 最坏存储一整层（完全图中 O(V)），DFS 最坏存储一整条路径（链状图中 O(V)）。在宽而浅的图中 DFS 更省内存，在深而窄的图中 BFS 更省内存。

## 常见陷阱

    - 错误地用 DFS 求无权图最短路径，DFS 不保证最短
    - BFS 在 JavaScript 中 `Array.shift()` 是 O(n)，大数据量应用双端队列
    - DFS 递归过深会栈溢出，需改用迭代实现



## 内存使用详细分析

  BFS 在最坏情况下需要存储一整层的顶点。对于完全二叉树，最后一层有 n/2 个节点，所以 BFS 空间 O(n)。DFS 只需存储一条路径上的节点，对于深度为 d 的树只需 O(d)。在广而浅的图中 DFS 更省内存，在深而窄的图中 BFS 更省内存。

## 代码对比

```javascript
// BFS 模板
function bfs(graph, start) {
  const visited = new Set([start]);
  const queue = [start];
  while (queue.length) {
    const u = queue.shift();  // 注意: Array.shift 是 O(n)
    console.log('访问:', u);
    for (const v of (graph[u] || [])) {
      if (!visited.has(v)) {
        visited.add(v);
        queue.push(v);
      }
    }
  }
}

// DFS 模板
function dfs(graph, start) {
  const visited = new Set();
  function go(u) {
    visited.add(u);
    console.log('访问:', u);
    for (const v of (graph[u] || [])) {
      if (!visited.has(v)) go(v);
    }
  }
  go(start);
}
```

## LeetCode 应用指南

  | 题目类型 | 推荐算法 | 原因 |
  | --- | --- | --- |
  | 最短路径（无权） | BFS | 逐层扩展保证最短 |
  | 所有路径枚举 | DFS | 回溯天然适合穷举 |
  | 连通分量 | 两者皆可 | BFS/DFS 均可标记 |
  | 拓扑排序 | BFS(Kahn)/DFS | 两种方法都行 |
  | 层序遍历 | BFS | 天然按层处理 |
  | 岛屿问题 | DFS | 递归简洁 |

## 实际场景选择

  - **社交网络推荐好友：**用 BFS 从用户出发找 N 度好友（按距离分层）
  - **迷宫最短路：**BFS，保证最先到达终点的路径最短
  - **文件目录递归遍历：**DFS，天然的递归结构
  - **拼单词游戏：**DFS 回溯尝试所有字母组合

##交互演示
