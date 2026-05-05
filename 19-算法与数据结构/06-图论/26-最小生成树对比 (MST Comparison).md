## MST Comparison


```javascript
对比 Kruskal 和 Prim 算法的时间复杂度和适用场景。```


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
console.log('Prim: 优先队列扩展');```


  点击按钮查看结果
