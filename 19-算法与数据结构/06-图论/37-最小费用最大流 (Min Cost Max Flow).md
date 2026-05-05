## Min Cost Max Flow


```javascript
在最大流的基础上最小化总费用，用 SPFA 或 Dijkstra 找最短路增广。```


```
function minCostMaxFlow(capacity, cost, source, sink) {
  const n = capacity.length;
  const graph = Array.from({length:n}, () => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (capacity[i][j]) graph[i].push(j);
  let flow = 0, costTotal = 0;
  const cap = capacity.map(r => [...r]);
  while (true) {
    const dist = new Array(n).fill(Infinity);
    const parent = new Array(n).fill(-1);
    const inq = new Array(n).fill(false);
    dist[source] = 0; const q = [source]; inq[source] = true;
    while (q.length) {
      const u = q.shift(); inq[u] = false;
      for (const v of graph[u]) {
        if (cap[u][v] > 0 && dist[u] + cost[u][v] < dist[v]) {
          dist[v] = dist[u] + cost[u][v]; parent[v] = u;
          if (!inq[v]) { q.push(v); inq[v] = true; }
        }
      }
    }
    if (parent[sink] === -1) break;
    let add = Infinity;
    for (let v = sink; v !== source; v = parent[v]) add = Math.min(add, cap[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) { cap[parent[v]][v] -= add; cap[v][parent[v]] += add; }
    flow += add; costTotal += add * dist[sink];
  }
  return { flow, costTotal };
}
console.log('最小费用最大流');```


  点击按钮查看结果
