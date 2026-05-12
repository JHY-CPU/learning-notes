# 拓扑排序基础 (Topological Sort Basics)

  拓扑排序是有向无环图（DAG）的线性排序，每个顶点只出现在所有依赖之后。

## 什么是拓扑排序

  拓扑排序将 DAG 的所有顶点排成一个线性序列，使得对于任意有向边 (u, v)，u 在序列中出现在 v 之前。它本质上是将偏序关系扩展为线性（全序）关系。只有无环有向图（DAG）才有拓扑排序。

## 关键性质

    - DAG 可能有多个合法的拓扑排序，不唯一
    - 存在环的图没有拓扑排序
    - 入度为 0 的顶点可以排在序列最前面
    - Kahn 算法（BFS 法）是实现拓扑排序的主流方法之一

## 复杂度分析

    - **时间：**O(V+E)，计算入度 + BFS 遍历
    - **空间：**O(V)，入度数组和队列

## 适用场景 vs 替代方案

    - 有依赖关系的任务调度：拓扑排序给出合法执行顺序
    - 检测图是否有环：排序结果长度 < V 说明有环
    - DFS 法也能实现拓扑排序，两种方法各有优劣
    - 关键路径法（CPM）基于拓扑排序计算项目工期

## 常见陷阱

    - 有环的图返回的排序不包含所有顶点，需检查结果长度
    - 有多个合法排序时，不同实现可能返回不同结果
    - 忘记处理孤立顶点（入度和出度都为 0）


```
// Kahn 算法（BFS）
function topologicalSortKahn(graph, n) {
  const inDegree = new Array(n).fill(0);
  for (const u in graph)
    for (const v of graph[u]) inDegree[v]++;
  const q = [];
  for (let i = 0; i < n; i++) if (inDegree[i] === 0) q.push(i);
  const res = [];
  while (q.length) {
    const u = q.shift();
    res.push(u);
    for (const v of (graph[u] || [])) {
      inDegree[v]--;
      if (inDegree[v] === 0) q.push(v);
    }
  }
  return res.length === n ? res : []; // 有环返回空
}
console.log(topologicalSortKahn({0:[1,2],1:[3],2:[3],3:[]}, 4)); // [0,1,2,3] 或 [0,2,1,3]
```


  点击按钮查看结果
