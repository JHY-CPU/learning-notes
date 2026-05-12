# Dijkstra 算法基础 (Dijkstra Basics)

  Dijkstra 算法用于计算非负权图中单源最短路径。通过贪心策略，每次选择距离最短的未处理顶点。


```javascript
function minDist(dist, visited) {
  let min = Infinity, minV = null;
  for (const v in dist) {
    if (!visited.has(v) && dist[v] < min) {
      min = dist[v];
      minV = v;
    }
  }
  return minV;
}

function dijkstra(graph, start) {
  const dist = {}, visited = new Set();
  for (const v in graph) dist[v] = Infinity;
  dist[start] = 0;
  while (visited.size < Object.keys(graph).length) {
    const v = minDist(dist, visited);
    visited.add(v);
    for (const [w, weight] of Object.entries(graph[v])) {
      if (!visited.has(w)) {
        dist[w] = Math.min(dist[w], dist[v] + weight);
      }
    }
  }
  return dist;
}
```

## 核心思想


    - 维护从起点到各顶点的最短距离估计

    - 每次从未处理的顶点中选择距离最小的

    - 松弛该顶点的所有出边

    - 重复直到所有顶点都被处理



## 核心思想详解

  Dijkstra 算法本质上是一种贪心算法：每次从未确定最短距离的顶点中选择距离最小的，然后用它去更新邻居的距离。贪心的正确性基于：如果所有边权非负，那么当前距离最小的顶点不可能通过其他未处理顶点得到更短的距离。

## 正确性证明

  假设当前未处理中距离最小的顶点是 u，dist[u] 是最短距离。反证：如果存在更短路径经过某未处理顶点 v，则 dist[v] >= dist[u]（因为 u 最小），且 v 到 u 的边权非负，所以经过 v 的路径距离 >= dist[v] >= dist[u]，矛盾。

## 与 BFS 的关系

  BFS 是 Dijkstra 在所有边权为 1 时的特例。BFS 用队列按层扩展，Dijkstra 用优先队列按距离扩展。

## 局限性

  - 不能处理负权边：贪心假设"当前最小距离已确定"，负权边会破坏这一假设
  - 不能检测负权环：需要使用 Bellman-Ford
  - 全源最短路需要运行 V 次 Dijkstra，不如 Floyd 方便

## 实际应用

  - **GPS 导航：**Dijkstra/A* 实时计算最短路线
  - **网络路由：**OSPF 协议使用 Dijkstra 计算最短路径树
  - **游戏 AI：**NPC 寻路算法的基础
  - **社交网络：**计算两个人之间的最短关系链

## 交互演示
