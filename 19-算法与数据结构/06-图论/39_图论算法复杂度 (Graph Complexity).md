## Graph Complexity


```javascript
图论中常见算法的时间/空间复杂度总结。```


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
console.log('复杂度分析帮助选择合适的图算法');```


  点击按钮查看结果
