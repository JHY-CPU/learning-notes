# 最短路径概述 (Shortest Path Overview)

  在加权图中寻找两个顶点之间权重和最小的路径。根据图的特点不同，有多种算法。

## 什么是最短路径问题

  最短路径问题要求在加权图中找到从源点到目标点的边权重之和最小的路径。根据问题的规模和图的特性（是否有负权边、是否需要全源等），选择不同的算法。

## 核心概念

    - **松弛（Relaxation）：**检查是否能通过中间顶点缩短距离，是所有最短路算法的核心操作
    - **负权环：**环上边权和为负的环，经过它可以无限减小路径代价，最短路径无定义
    - **单源 vs 全源：**单源求一个起点到所有点的距离，全源求所有点对之间的距离

## 算法对比








  | 算法 | 时间复杂度 | 适用场景 |
| --- | --- | --- |
| BFS | O(V+E) | 无权图 |
| Dijkstra | O((V+E)logV) | 非负权图 |
| Bellman-Ford | O(VE) | 任意权(可负), 检测负环 |
| SPFA | O(kE) 平均 | 稀疏图, 负权边 |
| Floyd-Warshall | O(V³) | 全源最短路径 |

## 选择策略

    - 无权图：BFS，O(V+E)
    - 非负权单源：Dijkstra，O((V+E)logV)
    - 存在负权边：Bellman-Ford 或 SPFA
    - 全源最短路：Floyd-Warshall，O(V^3)
    - DAG 上最短路：拓扑排序 + 松弛，O(V+E)

## 常见陷阱

    - 在有权图中错误使用 BFS，BFS 只保证无权图最短路
    - Dijkstra 不能处理负权边，贪心策略失效
    - 负权环存在时，最短路径可能不存在（可无限减小）

## 实际应用

  GPS 导航系统中，道路有通行时间和距离等权重，Dijkstra/A* 算法实时计算最短路线。网络路由中，OSPF 协议使用 Dijkstra 算法计算最短路径树，确定数据包的最优转发路径。

## 松弛操作详解

```javascript
// 松弛（Relaxation）是所有最短路算法的核心
// 如果 dist[u] + w(u,v) < dist[v]，则更新 dist[v]
function relax(dist, u, v, weight) {
  if (dist[u] + weight < dist[v]) {
    dist[v] = dist[u] + weight;
    return true;  // 发生了更新
  }
  return false;
}
```

## DAG 上最短路

```javascript
// DAG 上最短路：拓扑排序 + 松弛，O(V+E)
function shortestPathDAG(adj, weights, n, src) {
  // 先拓扑排序
  const visited = new Array(n).fill(false);
  const stack = [];
  function dfs(u) {
    visited[u] = true;
    for (const v of (adj[u] || [])) if (!visited[v]) dfs(v);
    stack.push(u);
  }
  for (let i = 0; i < n; i++) if (!visited[i]) dfs(i);

  const dist = new Array(n).fill(Infinity);
  dist[src] = 0;
  while (stack.length) {
    const u = stack.pop();
    if (dist[u] === Infinity) continue;
    for (const v of (adj[u] || [])) {
      dist[v] = Math.min(dist[v], dist[u] + weights[u][v]);
    }
  }
  return dist;
}
```

## 路径还原

```javascript
// 记录前驱节点，还原最短路径
function restorePath(parent, target) {
  const path = [];
  for (let v = target; v !== -1; v = parent[v]) {
    path.push(v);
  }
  return path.reverse();
}
```

## LeetCode 相关题目

  - 743. 网络延迟时间（Dijkstra）
  - 787. K 站中转内最便宜的航班（Bellman-Ford）
  - 1334. 阈值距离内最少的城市数（Floyd）
  - 1514. 概率最大的路径（Dijkstra 变体）
  - 1928. 规定时间内到达终点的最小花费

## 交互演示

  选择一个起点，计算到各顶点的最短路径（将用后续算法实现）
