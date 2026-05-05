## Topological Sort Basics


```javascript
拓扑排序是有向无环图的线性排序，每个顶点只出现在所有依赖之后。```


```
// Kahn 算法（BFS）
function topologicalSortKahn(graph, n) {
  const inDegree = new Array(n).fill(0);
  for (const u in graph)
    for (const v of graph[u]) inDegree[v]++;
  const q = [];
  for (let i = 0; i < n; i++) if (inDegree[i] === 0) q.push(i);
  const res = [];
  while (q.length) {
    const u = q.shift();
    res.push(u);
    for (const v of (graph[u] || [])) {
      inDegree[v]--;
      if (inDegree[v] === 0) q.push(v);
    }
  }
  return res.length === n ? res : []; // 有环返回空
}
console.log(topologicalSortKahn({0:[1,2],1:[3],2:[3],3:[]}, 4)); // [0,1,2,3] 或 [0,2,1,3]```


  点击按钮查看结果
