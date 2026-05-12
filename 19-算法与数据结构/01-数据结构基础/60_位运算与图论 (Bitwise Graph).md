# Bitwise Graph

### 位运算在图论中的应用

用位掩码表示图的邻接关系，可以实现 O(1) 的邻接判断，并利用 CPU 的位运算指令加速传递闭包等图算法。

### 关键特性

- **位邻接矩阵**：g[u] 的第 v 位为 1 表示 u 到 v 有边
- **邻接判断**：O(1) 查询两节点是否有边
- **邻居枚举**：通过位操作提取所有邻居
- **传递闭包**：用 Floyd-Warshall + Bitset 加速，复杂度 O(n³/w)

### 时间与空间复杂度

| 操作 | 位运算优化 | 普通矩阵 |
|------|-----------|---------|
| 邻接判断 | O(1) | O(1) |
| 邻居枚举 | O(degree) | O(n) |
| 存储空间 | O(n²/w) | O(n²) |
| 传递闭包 | O(n³/w) | O(n³) |

w 为字长（32 或 64）。

### 适用场景 vs 替代方案

- **稠密图**：位矩阵比邻接表更紧凑
- **小规模图**：节点数不超过 64 时效果最好
- **传递闭包**：Bitset 优化可提速 32-64 倍
- **替代**：稀疏图用邻接表更省空间

### 常见陷阱

- 节点数超过位宽时需要使用 BigInt 或数组模拟
- JavaScript 的 32 位整数限制影响大规模图运算
- 无向图需要同时设置两个方向的位

```
// 位运算邻接矩阵
function buildGraph(n, edges) {
  const g = new Array(n).fill(0);
  for (const [u,v] of edges) {
    g[u] |= (1 << v);
    g[v] |= (1 << u);
  }
  return g;
}
function hasEdge(g, u, v) { return (g[u] & (1 << v)) !== 0; }
function neighbors(g, u, n) {
  const res = [];
  for (let i = 0; i < n; i++) if (g[u] & (1 << i)) res.push(i);
  return res;
}
const g = buildGraph(4, [[0,1],[1,2],[2,3]]);
console.log(hasEdge(g, 0, 1)); // true
console.log(neighbors(g, 1, 4)); // [0,2]
```


### 实际应用

- **竞争编程**：节点数 <= 60 的图问题用 Bitset 优化
- **编译器优化**：用位向量表示基本块间的可达性
- **网络分析**：小规模社交网络中快速计算传递闭包
- **路由算法**：用位矩阵快速计算最短路径的中间节点

  点击按钮查看结果
