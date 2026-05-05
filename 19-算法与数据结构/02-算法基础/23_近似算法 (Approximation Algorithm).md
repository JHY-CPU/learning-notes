## Approximation Algorithm


```javascript
近似算法在多项式时间内给出近似最优解，常用于 NP-hard 问题。```


```
// 顶点覆盖的近似算法（2-近似）
function approxVertexCover(graph, n) {
  const visited = new Array(n).fill(false);
  const cover = [];
  for (let u = 0; u < n; u++) {
    for (const v of (graph[u] || [])) {
      if (!visited[u] && !visited[v]) {
        visited[u] = visited[v] = true;
        cover.push(u, v);
      }
    }
  }
  return [...new Set(cover)];
}
const graph = {0:[1,2],1:[0,3],2:[0,3],3:[1,2]};
console.log(approxVertexCover(graph, 4));
// 2-近似：覆盖大小 ≤ 2 * 最优覆盖大小```


  点击按钮查看结果
