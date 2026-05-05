## DFS Connected Components


```javascript
DFS 可以找出图中的所有连通分量，每次 DFS 遍历一个连通分量。```


```
function connectedComponents(graph, n) {
  const visited = new Array(n).fill(false);
  const components = [];
  function dfs(u, comp) {
    visited[u] = true;
    comp.push(u);
    for (const v of (graph[u] || []))
      if (!visited[v]) dfs(v, comp);
  }
  for (let i = 0; i < n; i++) {
    if (!visited[i]) {
      const comp = [];
      dfs(i, comp);
      components.push(comp);
    }
  }
  return components;
}
const graph = {0:[1],1:[0,2],2:[1],3:[4],4:[3]};
console.log(connectedComponents(graph, 5));
// [[0,1,2],[3,4]]```


  点击按钮查看结果
