## Bipartite Graph


```javascript
二分图可以用两种颜色着色，使相邻顶点颜色不同，DFS 染色法判定。```


```
function isBipartite(graph) {
  const color = new Array(graph.length).fill(-1);
  function dfs(u, c) {
    color[u] = c;
    for (const v of (graph[u] || [])) {
      if (color[v] === c) return false;
      if (color[v] === -1 && !dfs(v, 1-c)) return false;
    }
    return true;
  }
  for (let i = 0; i < graph.length; i++)
    if (color[i] === -1 && !dfs(i, 0)) return false;
  return true;
}
console.log(isBipartite([[1,3],[0,2],[1,3],[0,2]])); // true
console.log(isBipartite([[1,2,3],[0,2],[0,1,3],[0,2]])); // false```


  点击按钮查看结果
