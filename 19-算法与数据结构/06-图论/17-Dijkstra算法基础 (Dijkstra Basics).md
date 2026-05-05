## Dijkstra 算法

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
}```

  ## 核心思想


    - 维护从起点到各顶点的最短距离估计

    - 每次从未处理的顶点中选择距离最小的

    - 松弛该顶点的所有出边

    - 重复直到所有顶点都被处理



  ## 交互演示
