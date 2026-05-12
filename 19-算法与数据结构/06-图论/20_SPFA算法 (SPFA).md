# SPFA 算法

  Shortest Path Faster Algorithm，Bellman-Ford 的队列优化版本。平均时间复杂度 O(kE)，k 通常小于 2。

## 什么是 SPFA

  SPFA 是对 Bellman-Ford 算法的优化。Bellman-Ford 每轮对所有边做松弛，但实际只有距离被更新的顶点的出边才可能继续松弛。SPFA 用队列维护"距离被更新过的"顶点，只松弛这些顶点的出边，大幅减少了不必要的操作。

## 关键性质

    - 平均时间复杂度 O(kE)，k 通常很小（2 左右）
    - 最坏情况下退化为 O(VE)，与 Bellman-Ford 相同
    - 可以处理负权边和检测负环
    - 通过记录每个顶点入队次数来检测负环（入队 V 次则有负环）

## 复杂度分析

    - **平均时间：**O(kE)，k 为每个顶点平均入队次数
    - **最坏时间：**O(VE)，特殊构造的图会触发最坏情况
    - **空间：**O(V+E)

## 适用场景 vs Bellman-Ford

    - 稀疏图中 SPFA 通常比 Bellman-Ford 快很多
    - 竞赛中 SPFA 是处理负权边的常用选择
    - 某些特殊图上 SPFA 可能被卡成 O(VE)，此时需谨慎
    - SPFA 也可以用于非负权图替代 Dijkstra，但不推荐

## 常见陷阱

    - SPFA 在竞赛中可能被恶意数据卡成最坏复杂度
    - 队列实现可以用普通数组代替，但双端队列 SLF 优化可以提速
    - 负环检测时入队次数阈值应为 V（不是 V-1）


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
}
```

## 实际应用

  在网络流量工程中，SPFA 用于计算最短路径并适应链路权重的动态变化。在差分约束系统中，将约束条件转化为图的边，SPFA 求解可行性。
