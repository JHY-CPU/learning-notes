# DFS 连通分量 (DFS Connected Components)

  DFS 可以找出图中的所有连通分量，每次 DFS 遍历一个连通分量。

## 什么是连通分量

  在无向图中，连通分量是顶点的最大子集，其中任意两个顶点之间都存在路径。如果图不连通，则包含多个连通分量。DFS 是求连通分量的最直观方法：从未访问的顶点启动 DFS，访问到的所有顶点构成一个连通分量。

## 关键性质

    - 每次从未访问顶点启动 DFS，恰好覆盖一个连通分量
    - 连通分量数量 = 启动 DFS 的次数
    - 对于有向图，连通分量的概念需扩展为"强连通分量"（SCC）
    - 并查集（Union-Find）也可以求连通分量，代码更简洁

## 复杂度分析

    - **时间：**O(V+E)，每个顶点和每条边各访问一次
    - **空间：**O(V)，visited 数组和递归栈
    - 并查集方法时间复杂度为 O(E * alpha(V))，alpha 为反阿克曼函数

## 适用场景 vs 并查集

    - 需要列出每个连通分量的成员：DFS 更直观
    - 只需判断两点是否连通或合并集合：并查集更高效
    - 动态加边的连通性查询：并查集是标准选择
    - 需要连通分量的其他信息（如大小、边界）：DFS 更灵活

## 常见陷阱

    - 有向图不能直接用此方法求强连通分量
    - 忘记遍历所有顶点，只启动一次 DFS，遗漏了其他连通分量
    - 递归深度过大时应改用迭代 DFS 或并查集

```
function connectedComponents(graph, n) {
  const visited = new Array(n).fill(false);
  const components = [];
  function dfs(u, comp) {
    visited[u] = true;
    comp.push(u);
    for (const v of (graph[u] || []))
      if (!visited[v]) dfs(v, comp);
  }
  for (let i = 0; i < n; i++) {
    if (!visited[i]) {
      const comp = [];
      dfs(i, comp);
      components.push(comp);
    }
  }
  return components;
}
const graph = {0:[1],1:[0,2],2:[1],3:[4],4:[3]};
console.log(connectedComponents(graph, 5));
// [[0,1,2],[3,4]]
```

## 实际应用

  在社交网络中，找出互为好友的用户群体（每个连通分量是一个社交圈子）。在图像处理中，标记连通的像素区域（连通域分析），用于目标检测和分割。
