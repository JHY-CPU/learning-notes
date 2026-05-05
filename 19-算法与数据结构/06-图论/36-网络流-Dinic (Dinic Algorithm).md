## Dinic Algorithm


```javascript
Dinic 算法用分层图+阻塞流，O(V²E) 时间求最大流，实际性能远优于理论界。```


```
function dinic(capacity, source, sink) {
  const n = capacity.length;
  const graph = Array.from({length:n}, () => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (capacity[i][j] > 0) graph[i].push(j);
  const edges = capacity.map(r => [...r]);
  function bfs() {
    const level = new Array(n).fill(-1);
    const q = [source]; level[source] = 0;
    while (q.length) { const u = q.shift(); for (const v of graph[u]) if (level[v] === -1 && edges[u][v] > 0) { level[v] = level[u] + 1; q.push(v); } }
    return level[sink] !== -1;
  }
  const it = new Array(n).fill(0);
  function dfs(u, f, level) {
    if (u === sink) return f;
    for (let i = it[u]; i < graph[u].length; i++) {
      const v = graph[u][i];
      if (level[v] === level[u]+1 && edges[u][v] > 0) {
        const minF = dfs(v, Math.min(f, edges[u][v]), level);
        if (minF > 0) { edges[u][v] -= minF; edges[v][u] += minF; return minF; }
      }
      it[u]++;
    }
    return 0;
  }
  let flow = 0;
  while (bfs()) { it.fill(0); while (true) { const f = dfs(source, Infinity, [...Array(n).fill(-1)]); if (!f) break; flow += f; } }
  return flow;
}
console.log('Dinic O(V²E)');```


  点击按钮查看结果
