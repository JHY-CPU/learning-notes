## Multi-source BFS


```javascript
多源BFS从多个起点同时开始广度优先搜索，求解到最近起点的距离。```


```
function multiSourceBFS(graph, sources) {
  const n = graph.length;
  const dist = new Array(n).fill(Infinity);
  const q = [];
  for (const s of sources) { dist[s] = 0; q.push(s); }
  while (q.length) {
    const u = q.shift();
    for (const v of (graph[u] || [])) {
      if (dist[v] > dist[u] + 1) { dist[v] = dist[u] + 1; q.push(v); }
    }
  }
  return dist;
}
const graph = {0:[1],1:[0,2,3],2:[1],3:[1,4],4:[3]};
console.log(multiSourceBFS(graph, [0, 4]));
// 距离最近的起点: [0,1,2,1,0]```


  点击按钮查看结果
