## Graph Coloring


```javascript
图着色是给顶点分配颜色，使相邻顶点颜色不同。贪心算法可以达到 Δ+1 色。```


```
function greedyColoring(graph, n) {
  const colors = new Array(n).fill(-1);
  const used = new Array(n).fill(false);
  colors[0] = 0;
  for (let u = 1; u < n; u++) {
    used.fill(false);
    for (const v of (graph[u]||[]))
      if (colors[v] !== -1) used[colors[v]] = true;
    let c = 0;
    while (used[c]) c++;
    colors[u] = c;
  }
  return colors;
}
const graph = {0:[1,2],1:[0,2,3],2:[0,1,3],3:[1,2]};
console.log(greedyColoring(graph, 4));
// 最多使用 4 种颜色（实际 2 种足够）```


  点击按钮查看结果
