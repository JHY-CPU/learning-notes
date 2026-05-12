# Dinic Algorithm


```javascript
Dinic 算法用分层图+阻塞流，O(V²E) 时间求最大流，实际性能远优于理论界。```


```
function dinic(capacity, source, sink) {
  const n = capacity.length;
  const graph = Array.from({length:n}, () => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (capacity[i][j] > 0) graph[i].push(j);
  const edges = capacity.map(r => [...r]);
  function bfs() {
    const level = new Array(n).fill(-1);
    const q = [source]; level[source] = 0;
    while (q.length) { const u = q.shift(); for (const v of graph[u]) if (level[v] === -1 && edges[u][v] > 0) { level[v] = level[u] + 1; q.push(v); } }
    return level[sink] !== -1;
  }
  const it = new Array(n).fill(0);
  function dfs(u, f, level) {
    if (u === sink) return f;
    for (let i = it[u]; i < graph[u].length; i++) {
      const v = graph[u][i];
      if (level[v] === level[u]+1 && edges[u][v] > 0) {
        const minF = dfs(v, Math.min(f, edges[u][v]), level);
        if (minF > 0) { edges[u][v] -= minF; edges[v][u] += minF; return minF; }
      }
      it[u]++;
    }
    return 0;
  }
  let flow = 0;
  while (bfs()) { it.fill(0); while (true) { const f = dfs(source, Infinity, [...Array(n).fill(-1)]); if (!f) break; flow += f; } }
  return flow;
}
console.log('Dinic O(V²E)');```


## 算法原理

  Dinic 算法是效率最高的最大流算法之一，核心思想：
  1. **BFS 建层图：**计算每个顶点到源点的距离 level
  2. **DFS 找阻塞流：**只沿着 level[v] = level[u] + 1 的边搜索，使用当前弧优化
  3. **重复：**直到汇点不可达

  当前弧优化：DFS 中每条边最多被访问一次，跳过已经处理过的边。

## 单位容量图优化

  对于所有边容量为 1 的图（如二分图匹配），Dinic 的复杂度可优化到 O(E*sqrt(V))。

## 完整实现

```javascript
function dinic(capacity, source, sink) {
  const n = capacity.length;
  const graph = Array.from({length: n}, () => []);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++)
      if (capacity[i][j] > 0) graph[i].push(j);
  const edges = capacity.map(r => [...r]);

  function bfs() {
    const level = new Array(n).fill(-1);
    const q = [source]; level[source] = 0;
    while (q.length) {
      const u = q.shift();
      for (const v of graph[u]) {
        if (level[v] === -1 && edges[u][v] > 0) {
          level[v] = level[u] + 1;
          q.push(v);
        }
      }
    }
    return level[sink] !== -1 ? level : null;
  }

  function dfs(u, pushed, level) {
    if (u === sink) return pushed;
    for (let i = it[u]; i < graph[u].length; i++) {
      const v = graph[u][i];
      if (level[v] === level[u] + 1 && edges[u][v] > 0) {
        const flow = dfs(v, Math.min(pushed, edges[u][v]), level);
        if (flow > 0) {
          edges[u][v] -= flow;
          edges[v][u] += flow;
          return flow;
        }
      }
      it[u]++;
    }
    return 0;
  }

  let totalFlow = 0;
  let level;
  while ((level = bfs()) !== null) {
    const it = new Array(n).fill(0);
    while (true) {
      const pushed = dfs(source, Infinity, level);
      if (pushed === 0) break;
      totalFlow += pushed;
    }
  }
  return totalFlow;
}
```

## 应用场景

  - **大规模网络流：**Dinic 在实践中远优于理论复杂度
  - **二分图匹配：**Hopcroft-Karp 实际上是 Dinic 的特例
  - **最小路径覆盖：**转化为最大流
  - **多商品流问题：**分层图的思想可扩展

  点击按钮查看结果
