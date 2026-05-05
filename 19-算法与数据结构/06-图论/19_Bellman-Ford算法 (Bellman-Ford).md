## Bellman-Ford 算法

  Bellman-Ford 算法可以处理负权边，并能检测负环。时间复杂度 O(VE)。


```javascript
function bellmanFord(edges, V, start) {
  const dist = Array(V).fill(Infinity);
  dist[start] = 0;
  for (let i = 0; i < V-1; i++) {
    for (const [u,v,w] of edges) {
      if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
      }
    }
  }
  // 检测负环
  for (const [u,v,w] of edges) {
    if (dist[u] + w < dist[v]) {
      return null; // 存在负环
    }
  }
  return dist;
}```

  ## 交互演示
