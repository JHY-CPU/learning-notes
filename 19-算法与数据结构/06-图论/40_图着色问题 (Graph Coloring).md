# Graph Coloring

  图着色是给顶点分配颜色，使相邻顶点颜色不同。贪心算法可以达到 Δ+1 色。

## 什么是图着色

  图着色要求为图的每个顶点分配一种颜色，使得任意两个相邻顶点颜色不同。所需的最少颜色数称为图的色数（Chromatic Number）。图着色问题是 NP-完全的，但贪心算法提供了一个实用的近似。

## 关键性质

    - 四色定理：任何平面图可以用 4 种颜色着色
    - 贪心着色最多使用 Δ+1 种颜色（Δ 为最大度数）
    - Brooks 定理：连通图（非完全图和奇环）的色数 <= Δ
    - 色数 >= 团数（最大完全子图的顶点数）


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
// 最多使用 4 种颜色（实际 2 种足够）
```


## 复杂度分析

    - **贪心着色：**O(V+E)，处理每个顶点及其邻边
    - **精确最小着色：**NP-完全，O(2^V) 或更差
    - **空间：**O(V)，颜色数组

## 适用场景

    - 排课问题：课程是顶点，冲突课程是边，色数 = 所需最少时段
    - 寄存器分配：编译器中变量是顶点，同时活跃的变量连边
    - 频率分配：蜂窝网络中基站的频率分配

## 改进贪心着色（最大度优先）

```javascript
// 按度数降序排列顶点，通常能得到更好的结果
function degreeOrderColoring(graph, n) {
  // 按度数降序排序
  const order = Array.from({length: n}, (_, i) => i)
    .sort((a, b) => (graph[b] || []).length - (graph[a] || []).length);

  const colors = new Array(n).fill(-1);
  colors[order[0]] = 0;

  for (let idx = 1; idx < n; idx++) {
    const u = order[idx];
    const used = new Array(n).fill(false);
    for (const v of (graph[u] || [])) {
      if (colors[v] !== -1) used[colors[v]] = true;
    }
    let c = 0;
    while (used[c]) c++;
    colors[u] = c;
  }
  return { colors, numColors: Math.max(...colors) + 1 };
}
```

## 边着色

  边着色要求相邻边颜色不同。Vizing 定理：简单图的边色数为 Delta 或 Delta+1。

## 四色定理应用

  任何平面地图可用 4 种颜色着色，使相邻区域颜色不同。这是图着色最著名的应用。

## 常见陷阱

    - 贪心着色的结果取决于顶点处理顺序，不同顺序得到不同色数
    - 求最小色数是 NP-完全问题，大规模问题只能用近似
    - 边着色和顶点着色是不同问题，不要混淆
    - Welsh-Powell 算法（按度数降序着色）通常优于朴素贪心
