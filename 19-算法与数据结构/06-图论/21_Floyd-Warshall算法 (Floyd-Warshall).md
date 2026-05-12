# Floyd-Warshall 算法

  多源最短路径算法，使用动态规划思想，O(V^3) 计算所有顶点对之间的最短路径。

## 什么是 Floyd-Warshall

  Floyd-Warshall 是一种全源最短路径算法，通过三重循环枚举中间顶点 k，逐步更新任意两点之间的最短距离。其核心思想是：对于任意两点 i 和 j，检查是否存在一个中间顶点 k，使得经过 k 的路径更短。

## 关键性质

    - 基于动态规划：dist[k][i][j] 表示只允许经过前 k 个顶点时 i 到 j 的最短距离
    - 可以处理负权边，但不能有负权环
    - 空间优化：可以就地更新，省去 k 维度
    - 代码极其简洁，只有 5 行核心逻辑

## 复杂度分析

    - **时间：**O(V^3)，三重循环
    - **空间：**O(V^2)，邻接矩阵
    - 不适合 V > 2000 的图，V=2000 时约 80 亿次运算

## 适用场景 vs 单源算法

    - 需要所有点对之间的最短距离：Floyd 是唯一选择
    - 只需单源最短路：Dijkstra 或 Bellman-Ford 更快
    - 顶点数少（V <= 500）时 Floyd 简单可靠
    - 检测负环：检查 dist[i][i] 是否 < 0

## 常见陷阱

    - V 较大时（> 1000）O(V^3) 超时，应改用 V 次 Dijkstra
    - 初始化时不存在的边应设为 Infinity，不可遗漏
    - 循环顺序必须是 k-i-j，k 必须在最外层


```javascript
function floydWarshall(graph) {
  const V = graph.length;
  const dist = graph.map(row => [...row]);
  for (let k = 0; k < V; k++)
    for (let i = 0; i < V; i++)
      for (let j = 0; j < V; j++)
        if (dist[i][j] > dist[i][k] + dist[k][j])
          dist[i][j] = dist[i][k] + dist[k][j];
  return dist;
}
```

## DP 递推式

  dist[k][i][j] = min(dist[k-1][i][j], dist[k-1][i][k] + dist[k-1][k][j])

## 实际应用

  在城市间航班网络中（城市数通常 < 500），Floyd 预计算后可以 O(1) 查询任意两城市间的最短飞行代价。在传递闭包中，将 Floyd 的 min 改为 OR/AND，可以判断任意两点是否连通。
