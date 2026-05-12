# Edmonds-Karp


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


## 算法原理

  Edmonds-Karp 是 Ford-Fulkerson 方法的一个具体实现，每次用 BFS 找**最短**增广路（边数最少）。这保证了复杂度上界 O(VE^2)，优于 DFS 版 Ford-Fulkerson 的不确定上界。

## 正确性关键

  每条边最多成为关键边（增广路上残余容量最小的边）O(V) 次。因为每次成为关键边后，从 s 到该边的距离至少增加 2，而最远不超过 V，所以最多 V/2 次。

## 复杂度分析

  - **时间：**O(VE^2)，最多 O(E) 次增广，每次 BFS O(V+E)
  - **空间：**O(V^2)（邻接矩阵）或 O(V+E)（邻接表）

## 与 Dinic 的对比

  | 特性 | Edmonds-Karp | Dinic |
  | --- | --- | --- |
  | 搜索方式 | 每次 BFS 找最短路 | BFS 建层 + DFS 找阻塞流 |
  | 时间复杂度 | O(VE^2) | O(V^2E) |
  | 实际性能 | 一般 | 更快 |
  | 实现难度 | 简单 | 中等 |

## 应用场景

  - 小规模网络流问题
  - 教学演示最大流算法
  - 二分图匹配（转化最大流）

  点击按钮查看结果
