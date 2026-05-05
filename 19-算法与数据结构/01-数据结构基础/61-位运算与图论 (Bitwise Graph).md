## Bitwise Graph


```javascript
位运算在图论中用于快速邻接判断、传递闭包（Bitset 优化）。```


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
console.log(neighbors(g, 1, 4)); // [0,2]```


  点击按钮查看结果
