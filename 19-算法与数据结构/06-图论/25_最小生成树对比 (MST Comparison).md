# MST Comparison

  对比 Kruskal 和 Prim 算法的时间复杂度和适用场景。

## 核心区别

  Kruskal 是全局贪心（按边排序），Prim 是局部贪心（从顶点扩展）。两者都能正确求出 MST，但适用的图类型不同。

## 关键性质

    - 最小生成树的边权总和是所有生成树中最小的
    - MST 不唯一，但最小总权重唯一
    - 切割性质：对于任意切割，横跨切割的最小权边一定在 MST 中
    - 回路性质：环中最大权边一定不在 MST 中


```
// Kruskal vs Prim
// Kruskal: O(E log E) — 适合稀疏图
//   - 按边权重排序
//   - 用并查集检测环
//   - 全局贪心
// Prim: O(V²) 或 O(E log V) — 适合稠密图
//   - 从起点扩展
//   - 维护到已选集合的最小距离
//   - 局部贪心
//
// 选择建议:
// 边数少 → Kruskal
// 顶点少 → Prim (邻接矩阵 O(V²))
console.log('Kruskal: 边排序+并查集');
console.log('Prim: 优先队列扩展');
```


## 常见陷阱

    - MST 只适用于无向连通加权图
    - 有向图的最小生成树需要不同的算法（如 Chu-Liu/Edmonds）
    - 边权相同时 MST 可能不唯一，但总权重相同

## Kruskal 代码实现

```javascript
// Kruskal 算法
function kruskal(n, edges) {
  edges.sort((a, b) => a[2] - b[2]);  // 按权重排序

  // 并查集
  const parent = Array.from({length: n}, (_, i) => i);
  function find(x) { return parent[x] === x ? x : parent[x] = find(parent[x]); }
  function union(x, y) { parent[find(x)] = find(y); }

  let cost = 0;
  const mst = [];
  for (const [u, v, w] of edges) {
    if (find(u) !== find(v)) {
      union(u, v);
      cost += w;
      mst.push([u, v, w]);
    }
  }
  return { cost, edges: mst };
}
```

## Prim 代码实现

```javascript
// Prim 算法（优先队列版）
function prim(adj, n) {
  const visited = new Set();
  let cost = 0;
  // 简化：用数组模拟优先队列
  const edges = [[0, 0, 0]];  // [from, to, weight]
  visited.add(0);

  while (visited.size < n) {
    edges.sort((a, b) => b[2] - a[2]);
    const [from, to, w] = edges.pop();
    if (visited.has(to)) continue;
    visited.add(to);
    cost += w;
    for (const [v, wt] of (adj[to] || [])) {
      if (!visited.has(v)) edges.push([to, v, wt]);
    }
  }
  return cost;
}
```

## 详细对比

  | 特性 | Kruskal | Prim |
  | --- | --- | --- |
  | 策略 | 全局最小边 | 局部最小边 |
  | 数据结构 | 并查集 | 优先队列 |
  | 时间 | O(E log E) | O(E log V) |
  | 适合 | 稀疏图 | 稠密图 |
  | 需要 | 边列表 | 邻接表 |

## 实际应用

  - **网络布线：**城市间建设高速公路/光纤网络，最小化总建设成本
  - **聚类分析：**MST 去掉最大边可将图分成若干类（单链接聚类）
  - **电路设计：**PCB 布线中的最小总线长
  - **图像分割：**基于图的图像分割算法

## LeetCode 相关题目

  - 1135. 最低成本连通所有城市
  - 1584. 连接所有点的最小费用
  - 1489. 找到最小生成树的关键边和伪关键边
