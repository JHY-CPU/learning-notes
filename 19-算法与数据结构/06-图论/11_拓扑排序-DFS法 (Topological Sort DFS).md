## Topological Sort DFS


```javascript
DFS 完成后将节点入栈，逆序即为拓扑排序。```


```
function topologicalSortDFS(graph, n) {
  const visited = new Array(n).fill(false);
  const onPath = new Array(n).fill(false);
  const res = [];
  let hasCycle = false;
  function dfs(u) {
    if (onPath[u]) { hasCycle = true; return; }
    if (visited[u]) return;
    visited[u] = true;
    onPath[u] = true;
    for (const v of (graph[u] || [])) dfs(v);
    onPath[u] = false;
    res.push(u);
  }
  for (let i = 0; i < n; i++) if (!visited[i]) dfs(i);
  return hasCycle ? [] : res.reverse();
}
console.log(topologicalSortDFS({0:[1,2],1:[3],2:[3],3:[]}, 4));
// [0, 1, 2, 3] 或 [0, 2, 1, 3]```


  点击按钮查看结果
