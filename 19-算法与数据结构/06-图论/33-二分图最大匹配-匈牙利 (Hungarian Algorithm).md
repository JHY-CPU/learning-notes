## Hungarian Algorithm


```javascript
匈牙利算法在 O(VE) 时间内求二分图最大匹配。```


```
function hungarian(graph, n, m) {
  const match = new Array(m).fill(-1);
  function dfs(u, seen) {
    for (const v of (graph[u]||[])) {
      if (seen[v]) continue;
      seen[v] = true;
      if (match[v] === -1 || dfs(match[v], seen)) { match[v] = u; return true; }
    }
    return false;
  }
  let result = 0;
  for (let i = 0; i < n; i++) { const seen = new Array(m).fill(false); if (dfs(i, seen)) result++; }
  return { match, size: result };
}
const graph = {0:[0,1],1:[1,2],2:[2]};
console.log(hungarian(graph, 3, 3)); // size: 3```


  点击按钮查看结果
