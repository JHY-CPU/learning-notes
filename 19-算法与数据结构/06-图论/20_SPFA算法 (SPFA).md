## SPFA 算法

  Shortest Path Faster Algorithm，Bellman-Ford 的队列优化版本。平均时间复杂度 O(kE)，k 通常小于 2。


```javascript
function spfa(graph, start) {
  const dist = {}, inq = {}, queue = [start];
  for (const v in graph) dist[v] = Infinity;
  dist[start] = 0;
  inq[start] = true;
  while (queue.length) {
    const v = queue.shift();
    inq[v] = false;
    for (const [w, wt] of Object.entries(graph[v])) {
      if (dist[w] > dist[v] + wt) {
        dist[w] = dist[v] + wt;
        if (!inq[w]) { queue.push(w); inq[w] = true; }
      }
    }
  }
  return dist;
}```

  ## 交互演示
