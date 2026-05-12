# BFS 最短路径 (BFS Shortest Path)

  在无权图中，BFS 第一次访问到目标顶点时经过的路径就是最短路径（边数最少）。

## 为什么 BFS 能求最短路径

  BFS 按层遍历图，第 k 层的顶点到起点恰好经过 k 条边。当 BFS 首次到达目标顶点时，经过的边数一定是最少的。这是因为 BFS 保证在访问距离为 k+1 的顶点之前，已经访问了所有距离为 k 的顶点。

## 关键性质

    - 需要额外维护 `dist` 数组记录每个顶点到起点的距离
    - 维护 `prev` 数组（前驱数组），用于回溯重建完整路径
    - 仅适用于无权图或等权图（所有边权值相同）
    - 对有权图应使用 Dijkstra 或 Bellman-Ford

## 复杂度分析

    - **时间：**O(V+E)，与普通 BFS 相同
    - **空间：**O(V)，存储 dist 和 prev 数组
    - **路径重建：**O(路径长度)，最坏 O(V)

## 适用场景 vs 替代方案

    - 无权图或等权图的最短路径问题首选
    - 有权图必须使用 Dijkstra（非负权）或 Bellman-Ford（可负权）
    - 多源最短路径可以用多源 BFS（多个起点同时入队）
    - 网格图中的最短路径天然适合 BFS

## 常见陷阱

    - 有权图中 BFS 不保证最短路径，因为边权重不同
    - `dist` 数组未初始化为 `Infinity` 可能导致判断错误
    - 不维护 prev 数组则无法输出具体路径

```javascript
function shortestPath(graph, start, end) {
  const queue = [start];
  const dist = { [start]: 0 };
  const prev = { [start]: null };
  while (queue.length) {
    const v = queue.shift();
    if (v === end) break;
    for (const w of graph[v]) {
      if (dist[w] === undefined) {
        dist[w] = dist[v] + 1;
        prev[w] = v;
        queue.push(w);
      }
    }
  }
  // 回溯重建路径
  const path = [];
  let cur = end;
  while (cur !== null) {
    path.unshift(cur);
    cur = prev[cur];
  }
  return { path, dist: dist[end] };
}
```

## 实际应用

  在棋类游戏中，计算棋子从当前位置到目标位置的最少步数。在无权网络中，找到数据包从源节点到目的节点经过的最少跳数（hop count）。

  查找从 A 到 G 的最短路径
