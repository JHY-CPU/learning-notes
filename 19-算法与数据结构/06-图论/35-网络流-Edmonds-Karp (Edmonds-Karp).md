## Edmonds-Karp


```javascript
Edmonds-Karp 用 BFS 找最短增广路，时间复杂度 O(VE²)。```


```
// Edmonds-Karp 与 Ford-Fulkerson 类似，但总是用 BFS 找最短增广路
// 确保了 O(VE²) 的时间复杂度上界
function edmondsKarp(capacity, source, sink) {
  const n = capacity.length;
  const res = capacity.map(r => [...r]);
  let flow = 0;
  while (true) {
    const parent = new Array(n).fill(-1);
    const q = [source]; parent[source] = source;
    while (q.length && parent[sink] === -1) {
      const u = q.shift();
      for (let v = 0; v < n; v++)
        if (parent[v] === -1 && res[u][v] > 0) { parent[v] = u; q.push(v); }
    }
    if (parent[sink] === -1) break;
    let add = Infinity;
    for (let v = sink; v !== source; v = parent[v]) add = Math.min(add, res[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) { res[parent[v]][v] -= add; res[v][parent[v]] += add; }
    flow += add;
  }
  return flow;
}
console.log('Edmonds-Karp O(VE²)');```


  点击按钮查看结果
