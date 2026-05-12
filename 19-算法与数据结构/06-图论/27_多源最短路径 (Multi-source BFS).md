# Multi-source BFS

  多源BFS从多个起点同时开始广度优先搜索，求解到最近起点的距离。

## 什么是多源 BFS

  多源 BFS 是 BFS 的扩展，将多个源点同时加入队列作为起点。BFS 的逐层扩散特性保证了：对于任意顶点，它被首次访问时一定是由最近的源点到达的。多源 BFS 在许多实际问题中非常有用。

## 关键性质

    - 将所有源点同时入队并设距离为 0，然后执行标准 BFS
    - 与单源 BFS 时间复杂度相同，O(V+E)
    - 每个顶点记录的是到最近源点的距离，而非到特定源点的距离
    - 可以扩展为多源 Dijkstra（多个源点同时入优先队列）

## 复杂度分析

    - **时间：**O(V+E)，与单源 BFS 相同
    - **空间：**O(V)，dist 数组和队列


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
// 距离最近的起点: [0,1,2,1,0]
```


## 适用场景

    - 在网格中找到每个空格到最近障碍物/目标的距离
    - 疫情扩散模拟：多个感染源同时扩散
    - 多个基站的信号覆盖范围计算

## 多源 Dijkstra

```javascript
// 多源 Dijkstra：多个源点同时入优先队列
function multiSourceDijkstra(adj, sources) {
  const n = adj.length;
  const dist = new Array(n).fill(Infinity);
  const parent = new Array(n).fill(-1);
  const pq = [];  // [dist, node]

  for (const s of sources) {
    dist[s] = 0;
    pq.push([0, s]);
  }

  while (pq.length) {
    pq.sort((a, b) => a[0] - b[0]);
    const [d, u] = pq.shift();
    if (d > dist[u]) continue;
    for (const [v, w] of adj[u]) {
      if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
        parent[v] = u;
        pq.push([dist[v], v]);
      }
    }
  }
  return { dist, parent };
}
```

## LeetCode 经典例题

  - 994. 腐烂的橘子：多源 BFS，所有烂橘子同时开始扩散
  - 542. 01 矩阵：多源 BFS，所有 0 同时开始扩散
  - 1162. 地图分析：多源 BFS，求每个点到最近陆地的距离

```javascript
// LeetCode 994: 腐烂的橘子
function orangesRotting(grid) {
  const m = grid.length, n = grid[0].length;
  const queue = [];
  let fresh = 0;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] === 2) queue.push([i, j, 0]);
      else if (grid[i][j] === 1) fresh++;
    }
  }

  if (fresh === 0) return 0;
  const dirs = [[0,1],[0,-1],[1,0],[-1,0]];
  let maxTime = 0;

  while (queue.length) {
    const [r, c, t] = queue.shift();
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < m && nc >= 0 && nc < n && grid[nr][nc] === 1) {
        grid[nr][nc] = 2;
        fresh--;
        maxTime = t + 1;
        queue.push([nr, nc, t + 1]);
      }
    }
  }
  return fresh === 0 ? maxTime : -1;
}
```

## 网格图中的多源 BFS 模板

```javascript
function multiSourceBFSGrid(grid, sources, m, n) {
  const dist = Array.from({length: m}, () => new Array(n).fill(-1));
  const queue = [];

  for (const [r, c] of sources) {
    dist[r][c] = 0;
    queue.push([r, c]);
  }

  const dirs = [[0,1],[0,-1],[1,0],[-1,0]];
  while (queue.length) {
    const [r, c] = queue.shift();
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < m && nc >= 0 && nc < n && dist[nr][nc] === -1) {
        dist[nr][nc] = dist[r][c] + 1;
        queue.push([nr, nc]);
      }
    }
  }
  return dist;
}
```

## 常见陷阱

    - 源点未同时入队而分批入队，会导致距离计算错误
    - 多源 BFS 不记录每个顶点由哪个源点到达，需要额外数组
    - 网格图中注意边界检查和障碍物处理
    - 多源 Dijkstra 需要源点同时入优先队列
