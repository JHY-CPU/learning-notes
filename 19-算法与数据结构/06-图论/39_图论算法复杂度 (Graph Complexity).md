# Graph Complexity

  图论中常见算法的时间/空间复杂度总结。

## 为什么复杂度分析很重要

  图论算法的选择取决于输入规模和图的特性。V=100 时 O(V^3) 完全可以接受，但 V=10^5 时只能接受 O(V+E) 或 O(E log V) 的算法。理解复杂度是正确选型的前提。

## 复杂度分类

    - **线性级 O(V+E)：**BFS、DFS、拓扑排序、Tarjan SCC、割点/桥
    - **近线性级 O(E log V)：**Dijkstra（堆优化）、Prim（堆优化）
    - **准线性级 O(E log E)：**Kruskal（排序是瓶颈）
    - **多项式级 O(VE)：**Bellman-Ford、SPFA、匈牙利算法
    - **三次级 O(V^3)：**Floyd、Blossom 匹配


```
// 图论算法复杂度汇总
// 遍历: O(V+E)
// 拓扑排序: O(V+E)
// 最短路:
//   Dijkstra: O(V²) 或 O(E log V)
//   Bellman-Ford: O(VE)
//   Floyd: O(V³)
//   SPFA: O(kE) 平均，O(VE) 最坏
// 最小生成树:
//   Kruskal: O(E log E)
//   Prim: O(V²) 或 O(E log V)
// 强连通分量: O(V+E)
// 最大流:
//   Dinic: O(V²E)
//   对单位容量图: O(min(V^{2/3}, E^{1/2}) E)
// 二分图匹配: O(VE)
console.log('复杂度分析帮助选择合适的图算法');
```


## 实际应用

  在竞赛中，根据 V 和 E 的范围快速排除不合适的算法。V <= 200 用 Floyd，V <= 5000 用 Dijkstra/Bellman-Ford，V <= 10^6 用 BFS/DFS。在工程中，算法选择影响系统的响应时间和资源消耗。

## 详细复杂度对比表

  | 算法 | 最坏时间 | 平均/最好 | 空间 | 适用 V 范围 |
  | --- | --- | --- | --- | --- |
  | BFS/DFS | O(V+E) | O(V+E) | O(V) | V <= 10^7 |
  | Dijkstra(数组) | O(V^2) | O(V^2) | O(V) | V <= 5000 |
  | Dijkstra(堆) | O((V+E)logV) | O(ElogV) | O(V) | V <= 10^6 |
  | Bellman-Ford | O(VE) | O(VE) | O(V) | V <= 5000 |
  | SPFA | O(VE) | O(kE) | O(V) | V <= 10^5 |
  | Floyd | O(V^3) | O(V^3) | O(V^2) | V <= 500 |
  | Kruskal | O(ElogE) | O(ElogE) | O(V+E) | E <= 10^6 |
  | Prim(堆) | O(ElogV) | O(ElogV) | O(V+E) | V <= 10^6 |
  | Tarjan SCC | O(V+E) | O(V+E) | O(V) | V <= 10^7 |
  | Dinic | O(V^2E) | 远优理论 | O(V+E) | V <= 10^4 |
  | 匈牙利 | O(VE) | O(VE) | O(V) | V <= 5000 |

## 实际选型建议

```javascript
// 根据规模选算法的决策树
function chooseAlgorithm(V, E, hasNegative, needAllPairs) {
  if (needAllPairs) return V <= 500 ? 'Floyd' : '多次 Dijkstra';
  if (hasNegative) return 'Bellman-Ford / SPFA';
  if (V <= 5000) return 'Dijkstra(数组)';
  if (V <= 1000000) return 'Dijkstra(堆)';
  return 'BFS(无权) / 需优化';
}
```

## 竞赛快速判断

  - **V <= 200：**Floyd-Warshall 无脑用
  - **V <= 5000：**Dijkstra / Bellman-Ford
  - **V <= 10^5：**堆优化 Dijkstra / Kruskal
  - **V <= 10^6：**BFS/DFS / Tarjan
  - **V <= 10^7：**仅线性算法可用

## 常见陷阱

    - 混淆 V 和 E 的规模，稀疏图 E 约 V，稠密图 E 约 V^2
    - 忽略常数因子：O(V+E) 在稠密图上约等于 O(V^2)
    - SPFA 最坏 O(VE) 可能在竞赛中被卡
    - JavaScript 中 Object.keys() 遍历邻居会比数组慢
