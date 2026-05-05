## Prim MST


```javascript
Prim 算法从一个顶点开始，每次选择连接已选集合和未选集合的最小边。```


```
function prim(n, graph) {
  const visited = new Array(n).fill(false);
  const minDist = new Array(n).fill(Infinity);
  minDist[0] = 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    let u = -1;
    for (let j = 0; j < n; j++)
      if (!visited[j] && (u === -1 || minDist[j] < minDist[u])) u = j;
    if (minDist[u] === Infinity) break;
    visited[u] = true;
    total += minDist[u];
    for (const [v,w] of (graph[u] || []))
      if (!visited[v] && w < minDist[v]) minDist[v] = w;
  }
  return total;
}
const graph = {0:[[1,4],[2,3]],1:[[0,4],[2,1],[3,2]],2:[[0,3],[1,1],[3,4]],3:[[1,2],[2,4]]};
console.log(prim(4, graph)); // 6```


  点击按钮查看结果
