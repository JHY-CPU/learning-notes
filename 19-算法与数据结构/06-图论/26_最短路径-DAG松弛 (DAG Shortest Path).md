## DAG Shortest Path


```javascript
有向无环图的单源最短路径可通过拓扑顺序松弛实现 O(V+E)。```


```
function dagShortestPath(graph, n, start) {
  // 拓扑排序
  const inDegree = new Array(n).fill(0);
  for (const u in graph) for (const [v] of graph[u]) inDegree[v]++;
  const q = [];
  for (let i = 0; i < n; i++) if (inDegree[i] === 0) q.push(i);
  const topo = [];
  while (q.length) { const u = q.shift(); topo.push(u); for (const [v] of (graph[u]||[])) { inDegree[v]--; if (inDegree[v]===0) q.push(v); } }
  // 按拓扑序松弛
  const dist = new Array(n).fill(Infinity);
  dist[start] = 0;
  for (const u of topo)
    if (dist[u] !== Infinity)
      for (const [v,w] of (graph[u]||[]))
        if (dist[u] + w < dist[v]) dist[v] = dist[u] + w;
  return dist;
}
const dag = {0:[[1,5],[2,3]],1:[[3,6]],2:[[3,2],[4,4]],3:[[5,1]],4:[[5,2]],5:[]};
console.log(dagShortestPath(dag, 6, 0)); // [0,5,3,5,7,6]```


  点击按钮查看结果
