# 最短路径对比 (Shortest Path Comparison)

  选择正确的最短路径算法取决于图的特性和问题需求。以下是各算法的详细对比。





  | 算法 | 时间复杂度 | 空间 | 负权 | 负环检测 | 类型 |
| --- | --- | --- | --- | --- | --- |
| BFS | O(V+E) | O(V) | N/A(无权) | 否 | 单源 |
| Dijkstra | O((V+E)logV) | O(V) | 否 | 否 | 单源 |
| Bellman-Ford | O(VE) | O(V) | 是 | 是 | 单源 |
| SPFA | O(kE)~O(VE) | O(V) | 是 | 是 | 单源 |
| Floyd | O(V^3) | O(V^2) | 是 | 是 | 全源 |

## 选择指南


    - 无权图 -> BFS

    - 非负权 -> Dijkstra

    - 可能有负权 -> Bellman-Ford/SPFA

    - 需要所有点对 -> Floyd

## 复杂度总结

    - BFS：最快但仅限无权图
    - Dijkstra：非负权图的标准选择，堆优化版 O(E log V)
    - Bellman-Ford：处理负权边，O(VE) 较慢但可靠
    - SPFA：Bellman-Ford 的队列优化，平均快但最坏 O(VE)
    - Floyd：全源最短路 O(V^3)，适合小图

## 常见陷阱

    - 负权环存在时，Bellman-Ford/SPFA/Floyd 返回的结果无意义
    - 无权图不要用 Dijkstra，浪费性能
    - 竞赛中 SPFA 可能被卡，必要时回退 Bellman-Ford



## 各算法详解

  ### BFS（无权图）
  逐层扩展，第一次到达某顶点即为最短路。O(V+E)，简单高效。

  ### Dijkstra（非负权图）
  贪心选择当前距离最小的顶点。用优先队列优化后 O((V+E)logV)。不能处理负权边。

  ### Bellman-Ford（可负权）
  对所有边执行 V-1 轮松弛。O(VE)。第 V 轮仍有更新则存在负权环。

  ### SPFA（Bellman-Ford 优化）
  用队列维护需要松弛的顶点，平均 O(kE)。最坏仍 O(VE)，竞赛中可能被卡。

  ### Floyd（全源最短路）
  三重循环枚举中间点 k，O(V^3)。适合 V <= 500 的小图。

```javascript
// Floyd-Warshall 全源最短路
function floyd(dist, n) {
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (dist[i][k] + dist[k][j] < dist[i][j]) {
          dist[i][j] = dist[i][k] + dist[k][j];
        }
      }
    }
  }
  // 检查负环：对角线元素 < 0
  for (let i = 0; i < n; i++) {
    if (dist[i][i] < 0) return null;  // 存在负权环
  }
  return dist;
}
```

## 竞赛选型速查

  | V 范围 | 推荐算法 |
  | --- | --- |
  | V <= 500 | Floyd O(V^3) |
  | V <= 10000 | Dijkstra O(ElogV) |
  | V <= 10^6 | BFS O(V+E) |
  | 有负权边 | Bellman-Ford/SPFA |

## 交互演示
