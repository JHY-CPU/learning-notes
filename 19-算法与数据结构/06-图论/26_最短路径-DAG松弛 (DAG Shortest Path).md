# DAG Shortest Path

  有向无环图的单源最短路径可通过拓扑顺序松弛实现 O(V+E)。

## 为什么 DAG 可以线性时间求最短路

  DAG 没有环，因此存在拓扑排序。按照拓扑顺序依次松弛每个顶点的出边，可以保证处理顶点 u 时，所有指向 u 的边都已被处理，u 的最短距离已经确定。这避免了 Bellman-Ford 的重复松弛。

## 关键性质

    - 时间复杂度 O(V+E)，是最高效的单源最短路算法
    - 也可以处理负权边（DAG 中不存在环，不会出现负权环）
    - 同时可以求 DAG 的最长路径（将权重取反即可）
    - 依赖于拓扑排序的结果


```
function dagShortestPath(graph, n, start) {
  // 拓扑排序
  const inDegree = new Array(n).fill(0);
  for (const u in graph) for (const [v] of graph[u]) inDegree[v]++;
  const q = [];
  for (let i = 0; i < n; i++) if (inDegree[i] === 0) q.push(i);
  const topo = [];
  while (q.length) { const u = q.shift(); topo.push(u); for (const [v] of (graph[u]||[])) { inDegree[v]--; if (inDegree[v]===0) q.push(v); } }
  // 按拓扑序松弛
  const dist = new Array(n).fill(Infinity);
  dist[start] = 0;
  for (const u of topo)
    if (dist[u] !== Infinity)
      for (const [v,w] of (graph[u]||[]))
        if (dist[u] + w < dist[v]) dist[v] = dist[u] + w;
  return dist;
}
const dag = {0:[[1,5],[2,3]],1:[[3,6]],2:[[3,2],[4,4]],3:[[5,1]],4:[[5,2]],5:[]};
console.log(dagShortestPath(dag, 6, 0)); // [0,5,3,5,7,6]
```


## 适用场景 vs 其他算法

    - 确认是 DAG 时，此方法比 Dijkstra 和 Bellman-Ford 都快
    - 需要同时求拓扑排序和最短路时，一次完成
    - 关键路径法（最长路）的计算基础

## 常见陷阱

    - 必须先确认图是 DAG，有环时拓扑排序会失败
    - 松弛时只处理距离不为 Infinity 的顶点，避免无效操作
    - DAG 的最长路 = 将所有边权取反后求最短路，再取反
