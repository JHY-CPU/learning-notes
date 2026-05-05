## BFS 最短路径（无权图）

  在无权图中，BFS 第一次访问到目标顶点时经过的路径就是最短路径（边数最少）。


```javascript
function shortestPath(graph, start, end) {
  const queue = [start];
  const dist = { [start]: 0 };
  const prev = { [start]: null };
  while (queue.length) {
    const v = queue.shift();
    if (v === end) break;
    for (const w of graph[v]) {
      if (dist[w] === undefined) {
        dist[w] = dist[v] + 1;
        prev[w] = v;
        queue.push(w);
      }
    }
  }
  // 回溯重建路径
  const path = [];
  let cur = end;
  while (cur !== null) {
    path.unshift(cur);
    cur = prev[cur];
  }
  return { path, dist: dist[end] };
}```

  ## 交互演示

  查找从 A 到 G 的最短路径
