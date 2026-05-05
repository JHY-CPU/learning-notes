## Articulation Points & Bridges


```javascript
Tarjan 算法可以找出无向图中的割点和桥（关键边）。```


```
function findBridges(n, edges) {
  const graph = Array.from({length:n}, () => []);
  for (let i = 0; i < edges.length; i++) {
    const [u,v] = edges[i];
    graph[u].push([v, i]); graph[v].push([u, i]);
  }
  let timer = 0;
  const tin = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const bridges = [];
  function dfs(u, pEdge) {
    tin[u] = low[u] = timer++;
    for (const [v, ei] of graph[u]) {
      if (ei === pEdge) continue;
      if (tin[v] !== -1) low[u] = Math.min(low[u], tin[v]);
      else { dfs(v, ei); low[u] = Math.min(low[u], low[v]); if (low[v] > tin[u]) bridges.push([u,v]); }
    }
  }
  for (let i = 0; i < n; i++) if (tin[i] === -1) dfs(i, -1);
  return bridges;
}
console.log(findBridges(5, [[0,1],[0,2],[1,2],[1,3],[3,4]]));
// [[1,3],[3,4]] 或类似的连接```


  点击按钮查看结果
