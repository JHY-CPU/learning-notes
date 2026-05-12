# Minimum Cycle

  Floyd 算法可在求最短路的同时求出有向图中的最小环。

## 什么是最小环

  最小环是指图中所有环中边权和最小的环。在有向图中，可以利用 Floyd-Warshall 算法在 O(V^3) 时间内求出最小环。核心思路是：在 Floyd 的第 k 轮中，对于任意 i 和 j，若 i->k 和 k->j 的边存在，且 i 到 j 已有路径，则 i->k->j->i 构成一个环。

## 关键性质

    - 无向图的最小环可以用 Floyd 的变体求解，需记录次短路
    - 有向图的最小环可以在 Floyd 过程中顺带计算
    - 最小环的长度 >= 3（自环和二元环特殊处理）
    - 也可以用枚举边 + Dijkstra 的方法，O(E * E log V)


```
function minCycle(n, graph) {
  const INF = Infinity;
  const dist = Array.from({length:n}, () => new Array(n).fill(INF));
  const mat = Array.from({length:n}, () => new Array(n).fill(INF));
  for (let i = 0; i < n; i++)
    for (const [j,w] of (graph[i]||[])) { dist[i][j] = w; mat[i][j] = w; }
  let min = INF;
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < k; i++)
      for (let j = 0; j < k; j++)
        if (dist[i][j] < INF && mat[i][k] < INF && mat[k][j] < INF)
          min = Math.min(min, dist[i][j] + mat[i][k] + mat[k][j]);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++)
        if (dist[i][k] + dist[k][j] < dist[i][j])
          dist[i][j] = dist[i][k] + dist[k][j];
  }
  return min;
}
console.log('Floyd 最小环算法');
```


## 复杂度分析

    - **时间：**O(V^3)，与 Floyd 本身相同
    - **空间：**O(V^2)，邻接矩阵

## 适用场景

    - 有向图中求最短环路
    - 网络中检测最小反馈环
    - 交通规划中找最短闭合路线

## 常见陷阱

    - 注意最小环至少包含 3 条边（或 2 条有向边形成 2-环）
    - 无向图求最小环需要修改 Floyd 的逻辑
    - 负权边存在时最小环可能为负（即负权环）
