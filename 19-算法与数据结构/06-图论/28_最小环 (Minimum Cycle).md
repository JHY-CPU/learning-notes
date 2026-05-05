## Minimum Cycle


```javascript
Floyd 算法可在求最短路的同时求出有向图中的最小环。```


```
function minCycle(n, graph) {
  const INF = Infinity;
  const dist = Array.from({length:n}, () => new Array(n).fill(INF));
  const mat = Array.from({length:n}, () => new Array(n).fill(INF));
  for (let i = 0; i < n; i++)
    for (const [j,w] of (graph[i]||[])) { dist[i][j] = w; mat[i][j] = w; }
  let min = INF;
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++)
        if (dist[i][j] < INF && mat[i][k] < INF && mat[k][j] < INF)
          min = Math.min(min, dist[i][j] + mat[i][k] + mat[k][j]);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (dist[i][k] + dist[k][j] < dist[i][j])
          dist[i][j] = dist[i][k] + dist[k][j];
  }
  return min;
}
console.log('Floyd 最小环算法');```


  点击按钮查看结果
