# Bellman-Ford 算法

  Bellman-Ford 算法可以处理负权边，并能检测负环。时间复杂度 O(VE)。

## 什么是 Bellman-Ford

  Bellman-Ford 是一种单源最短路径算法，通过反复对所有边进行松弛操作来逐步逼近最短距离。与 Dijkstra 不同，它可以处理负权边，并且能检测图中是否存在负权环。算法的核心思想是：经过 k 轮松弛后，找到了最多经过 k 条边的最短路径。

## 关键性质

    - 对所有边执行 V-1 轮松弛，每轮确保至少一个顶点的最短距离被确定
    - 第 V 轮如果仍有松弛发生，说明存在负权环
    - 可以处理负权边，但负权环导致最短路径无定义
    - 比 Dijkstra 慢，但适用范围更广

## 复杂度分析

    - **时间：**O(VE)，V-1 轮循环，每轮遍历所有 E 条边
    - **空间：**O(V)，dist 数组

## 适用场景 vs Dijkstra

    - 有负权边：必须用 Bellman-Ford（或 SPFA）
    - 非负权图：Dijkstra 更快
    - 需要检测负环：Bellman-Ford 是标准方法
    - 稀疏图中 SPFA（Bellman-Ford 的队列优化）通常更快

## 常见陷阱

    - V-1 轮后还需要额外一轮检测负环，不能省略
    - 负权环存在时，返回的结果没有意义，调用方需检查
    - 边列表格式需统一，避免遗漏或重复


```javascript
function bellmanFord(edges, V, start) {
  const dist = Array(V).fill(Infinity);
  dist[start] = 0;
  for (let i = 0; i < V-1; i++) {
    for (const [u,v,w] of edges) {
      if (dist[u] + w < dist[v]) {
        dist[v] = dist[u] + w;
      }
    }
  }
  // 检测负环
  for (const [u,v,w] of edges) {
    if (dist[u] + w < dist[v]) {
      return null; // 存在负环
    }
  }
  return dist;
}
```

## 实际应用

  在货币套利检测中，将汇率取对数取负作为边权，负权环对应可无限获利的套利路径。在网络数据包路由中，Bellman-Ford 用于 RIP 协议计算最短路径，处理链路代价变化。
