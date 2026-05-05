## SCC Kosaraju


```javascript
Kosaraju 算法通过两次 DFS 找出有向图的强连通分量。```


```
function kosaraju(graph, n) {
  const visited = new Array(n).fill(false);
  const order = [];
  function dfs1(u) {
    visited[u] = true;
    for (const v of (graph[u]||[])) if (!visited[v]) dfs1(v);
    order.push(u);
  }
  for (let i = 0; i < n; i++) if (!visited[i]) dfs1(i);
  // 反向图
  const rev = Array.from({length:n}, () => []);
  for (const u in graph) for (const v of graph[u]) rev[v].push(Number(u));
  const result = [];
  const visited2 = new Array(n).fill(false);
  function dfs2(u, comp) {
    visited2[u] = true; comp.push(u);
    for (const v of rev[u]) if (!visited2[v]) dfs2(v, comp);
  }
  for (let i = n-1; i >= 0; i--) {
    const u = order[i];
    if (!visited2[u]) { const comp = []; dfs2(u, comp); result.push(comp); }
  }
  return result;
}
const g = {0:[1],1:[2],2:[0,3],3:[4],4:[3]};
console.log(kosaraju(g, 5)); // [[0,2,1],[4,3]]```


  点击按钮查看结果
