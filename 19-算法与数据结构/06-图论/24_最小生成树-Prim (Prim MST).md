# Prim MST

  Prim 算法从一个顶点开始，每次选择连接已选集合和未选集合的最小边。

## 什么是 Prim 算法

  Prim 也是一种贪心算法求最小生成树。与 Kruskal 的"按边排序"策略不同，Prim 从一个起始顶点出发，逐步扩展已选顶点集合。每一步选择连接"已选集合"和"未选集合"的权重最小的边，将其对应的顶点加入已选集合。

## 关键性质

    - 类似 Dijkstra 的结构，维护每个未选顶点到已选集合的最小距离
    - 使用优先队列（最小堆）优化可以达到 O(E log V)
    - 邻接矩阵实现时 O(V^2)，适合稠密图
    - Prim 从顶点出发扩展，Kruskal 从边出发排序

## 复杂度分析

    - **朴素实现：**O(V^2)，每轮扫描所有顶点找最小距离
    - **堆优化：**O(E log V)，使用优先队列
    - **空间：**O(V)，minDist 数组和 visited 数组

## 适用场景 vs Kruskal

    - 稠密图（E 接近 V^2）：Prim 的 O(V^2) 比 Kruskal 的 O(E log E) 更快
    - 邻接矩阵表示：Prim 天然适合
    - 稀疏图：Kruskal 通常更快
    - 需要增量添加顶点：Prim 更直观

## 常见陷阱

    - 朴素实现中找最小值的循环不要遗漏未访问顶点
    - 图不连通时 minDist 为 Infinity，需提前终止
    - 堆优化版本中更新距离后需要重新入堆（懒删除策略）


```
function prim(n, graph) {
  const visited = new Array(n).fill(false);
  const minDist = new Array(n).fill(Infinity);
  minDist[0] = 0;
  let total = 0;
  for (let i = 0; i < n; i++) {
    let u = -1;
    for (let j = 0; j < n; j++)
      if (!visited[j] && (u === -1 || minDist[j] < minDist[u])) u = j;
    if (minDist[u] === Infinity) break;
    visited[u] = true;
    total += minDist[u];
    for (const [v,w] of (graph[u] || []))
      if (!visited[v] && w < minDist[v]) minDist[v] = w;
  }
  return total;
}
const graph = {0:[[1,4],[2,3]],1:[[0,4],[2,1],[3,2]],2:[[0,3],[1,1],[3,4]],3:[[1,2],[2,4]]};
console.log(prim(4, graph)); // 6
```


## 实际应用

  在电网规划中，Prim 算法帮助确定以最低建设成本将所有变电站连接成网络的方案。在图像分割中，最小生成树可用于像素聚类和区域合并。
