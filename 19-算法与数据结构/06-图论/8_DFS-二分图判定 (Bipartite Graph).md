# DFS 二分图判定 (Bipartite Graph)

  二分图可以用两种颜色着色，使相邻顶点颜色不同，DFS 染色法判定。

## 什么是二分图

  二分图（Bipartite Graph）是指顶点集可以划分为两个不相交的子集，使得每条边的两个端点分别属于不同子集。等价地说，二分图可以用两种颜色对顶点着色，使得相邻顶点颜色不同。判断二分图的经典方法是 DFS 或 BFS 染色法。

## 关键性质

    - 二分图不含奇数长度的环（奇环），这是充要条件
    - 染色法：从任意顶点开始，交替染色 0 和 1，若冲突则不是二分图
    - 树一定是二分图（无环）
    - 二分图是最大匹配、匈牙利算法等的前置条件

## 复杂度分析

    - **时间：**O(V+E)，DFS/BFS 遍历全图
    - **空间：**O(V)，color 数组和递归栈

## 适用场景 vs 替代方案

    - 判断任务分配、资源匹配问题是否有解：转化为二分图判定
    - BFS 染色法和 DFS 染色法效果相同，BFS 更不容易栈溢出
    - 完全二分图 K(n,m) 有 n*m 条边，注意稠密图的边数上限

## 常见陷阱

    - 图可能不连通，必须对每个连通分量分别染色
    - 有向图需先转为无向图再判断
    - 奇环是二分图的唯一"障碍"，含奇环的图一定不是二分图

```
function isBipartite(graph) {
  const color = new Array(graph.length).fill(-1);
  function dfs(u, c) {
    color[u] = c;
    for (const v of (graph[u] || [])) {
      if (color[v] === c) return false;
      if (color[v] === -1 && !dfs(v, 1-c)) return false;
    }
    return true;
  }
  for (let i = 0; i < graph.length; i++)
    if (color[i] === -1 && !dfs(i, 0)) return false;
  return true;
}
console.log(isBipartite([[1,3],[0,2],[1,3],[0,2]])); // true
console.log(isBipartite([[1,2,3],[0,2],[0,1,3],[0,2]])); // false
```

## 实际应用

  在任务分配中，将工人和任务分别作为两个顶点集合，能做某任务则连边。判断该图是否为二分图决定了能否完美分配。在考试安排中，课程和时间段构成二分图，判断是否可以无冲突排课。
