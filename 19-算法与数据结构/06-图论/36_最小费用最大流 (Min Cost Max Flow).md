# Min Cost Max Flow


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


## 算法原理

  最小费用最大流在最大流的基础上增加费用维度：每条边有容量和单位费用。目标是在保证流最大的前提下，使总费用最小。

  核心思想：每次找**单位费用最小**的增广路。用 SPFA（或 Dijkstra + 势函数）找费用最短路，然后沿此路径增广。

## 势函数优化（Dijkstra 版）

  SPFA 在最坏情况下退化为 O(VE)，可以用 Johnson 势函数将边权转为非负，然后用 Dijkstra 替代 SPFA，每次 O(ElogV)。

```javascript
// 最小费用最大流（Dijkstra + 势函数版）
function mcmf(capacity, cost, source, sink) {
  const n = capacity.length;
  const graph = Array.from({length: n}, () => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (capacity[i][j]) graph[i].push(j);

  const cap = capacity.map(r => [...r]);
  const pot = new Array(n).fill(0);  // 势函数
  let flow = 0, totalCost = 0;

  while (true) {
    // 用势函数修正后的 Dijkstra
    const dist = new Array(n).fill(Infinity);
    const parent = new Array(n).fill(-1);
    dist[source] = 0;
    const pq = [[0, source]];

    while (pq.length) {
      pq.sort((a, b) => a[0] - b[0]);
      const [d, u] = pq.shift();
      if (d > dist[u]) continue;
      for (const v of graph[u]) {
        const reducedCost = cost[u][v] + pot[u] - pot[v];
        if (cap[u][v] > 0 && dist[u] + reducedCost < dist[v]) {
          dist[v] = dist[u] + reducedCost;
          parent[v] = u;
          pq.push([dist[v], v]);
        }
      }
    }

    if (parent[sink] === -1) break;

    // 增广
    let add = Infinity;
    for (let v = sink; v !== source; v = parent[v])
      add = Math.min(add, cap[parent[v]][v]);
    for (let v = sink; v !== source; v = parent[v]) {
      cap[parent[v]][v] -= add;
      cap[v][parent[v]] += add;
    }
    flow += add;
    totalCost += add * (dist[sink] + pot[sink]);

    // 更新势函数
    for (let i = 0; i < n; i++) if (dist[i] < Infinity) pot[i] += dist[i];
  }
  return { flow, cost: totalCost };
}
```

## 应用场景

  - **运输问题：**最小成本将货物从仓库运到商店
  - **任务分配：**每对工人-任务有不同成本，求最小总成本分配
  - **网络优化：**带宽有价格差异时的最优路由
  - **航空调度：**航班座位分配最小化总成本

  点击按钮查看结果
