## SCC Tarjan


```javascript
Tarjan 算法用一次 DFS 和 lowlink 找出所有强连通分量。```


```
function tarjan(graph, n) {
  let index = 0;
  const idx = new Array(n).fill(-1);
  const low = new Array(n).fill(0);
  const onStack = new Array(n).fill(false);
  const stack = [];
  const scc = [];
  function dfs(u) {
    idx[u] = low[u] = index++;
    stack.push(u); onStack[u] = true;
    for (const v of (graph[u]||[])) {
      if (idx[v] === -1) { dfs(v); low[u] = Math.min(low[u], low[v]); }
      else if (onStack[v]) low[u] = Math.min(low[u], idx[v]);
    }
    if (low[u] === idx[u]) {
      const comp = [];
      while (true) { const w = stack.pop(); onStack[w] = false; comp.push(w); if (w === u) break; }
      scc.push(comp);
    }
  }
  for (let i = 0; i < n; i++) if (idx[i] === -1) dfs(i);
  return scc;
}
console.log(tarjan({0:[1],1:[2],2:[0,3],3:[4],4:[3]}, 5));
// [[0,2,1],[4,3]]```


  点击按钮查看结果
