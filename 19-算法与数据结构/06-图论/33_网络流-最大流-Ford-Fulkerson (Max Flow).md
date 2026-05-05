## Max Flow


```javascript
Ford-Fulkerson 方法通过增广路径不断找增广路求最大流。```


```
// 简化版 Ford-Fulkerson
function maxFlow(capacity, source, sink) {
  const n = capacity.length;
  const residual = capacity.map(r => [...r]);
  let flow = 0;
  function bfs(parent) {
    const visited = new Array(n).fill(false);
    const q = [source]; visited[source] = true; parent[source] = -1;
    while (q.length) {
      const u = q.shift();
      for (let v = 0; v < n; v++) {
        if (!visited[v] && residual[u][v] > 0) {
          parent[v] = u; visited[v] = true; q.push(v);
        }
      }
    }
    return visited[sink];
  }
  const parent = new Array(n);
  while (bfs(parent)) {
    let pathFlow = Infinity;
    for (let v = sink; v !== source; v = parent[v]) pathFlow = Math.min(pathFlow, residual[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) { residual[parent[v]][v] -= pathFlow; residual[v][parent[v]] += pathFlow; }
    flow += pathFlow;
  }
  return flow;
}
const cap = [[0,16,13,0,0,0],[0,0,10,12,0,0],[0,4,0,0,14,0],[0,0,9,0,0,20],[0,0,0,7,0,4],[0,0,0,0,0,0]];
console.log(maxFlow(cap, 0, 5)); // 23```


  点击按钮查看结果
