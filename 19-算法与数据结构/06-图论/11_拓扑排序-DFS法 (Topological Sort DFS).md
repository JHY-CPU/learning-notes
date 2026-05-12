# 拓扑排序 - DFS法 (Topological Sort DFS)

  DFS 完成后将节点入栈，逆序即为拓扑排序。

## DFS 法原理

  对 DAG 执行 DFS，当一个顶点的所有后继都处理完毕后（即 DFS 返回时），将该顶点压入栈。最终栈中从顶到底就是拓扑排序序列。这利用了"后完成的依赖少"这一性质。

## 关键性质

    - 使用 `onPath` 数组记录当前 DFS 路径，用于检测环
    - 若 DFS 过程中遇到 onPath 中的顶点，说明存在环
    - DFS 法的结果顺序通常与 Kahn 算法不同，但都是合法排序
    - DFS 法天然能同时检测环和生成排序

## 复杂度分析

    - **时间：**O(V+E)，每个顶点和边各访问一次
    - **空间：**O(V)，visited、onPath、结果数组和递归栈

## 适用场景 vs Kahn 算法

    - DFS 法代码更紧凑，环检测内建于遍历过程
    - Kahn 算法更直观，便于理解入度递减的过程
    - 需要字典序最小的拓扑排序时，Kahn 配合优先队列更方便
    - 两种方法时间复杂度相同

## 常见陷阱

    - 忘记 `onPath` 标记的恢复（递归返回后设为 false），导致误判环
    - 有向图有环时返回空数组，调用方需检查
    - 递归深度过大时应改用迭代实现


```
function topologicalSortDFS(graph, n) {
  const visited = new Array(n).fill(false);
  const onPath = new Array(n).fill(false);
  const res = [];
  let hasCycle = false;
  function dfs(u) {
    if (onPath[u]) { hasCycle = true; return; }
    if (visited[u]) return;
    visited[u] = true;
    onPath[u] = true;
    for (const v of (graph[u] || [])) dfs(v);
    onPath[u] = false;
    res.push(u);
  }
  for (let i = 0; i < n; i++) if (!visited[i]) dfs(i);
  return hasCycle ? [] : res.reverse();
}
console.log(topologicalSortDFS({0:[1,2],1:[3],2:[3],3:[]}, 4));
// [0, 1, 2, 3] 或 [0, 2, 1, 3]
```


## Kahn 算法对比

```javascript
// Kahn 算法（BFS 法）实现
function topologicalSortKahn(graph, n) {
  const inDegree = new Array(n).fill(0);
  for (const u in graph) {
    for (const v of (graph[u] || [])) {
      inDegree[v]++;
    }
  }
  const queue = [];
  for (let i = 0; i < n; i++) if (inDegree[i] === 0) queue.push(i);

  const res = [];
  while (queue.length) {
    const u = queue.shift();
    res.push(u);
    for (const v of (graph[u] || [])) {
      if (--inDegree[v] === 0) queue.push(v);
    }
  }
  return res.length === n ? res : [];  // 长度不足说明有环
}
```

## 两种方法对比

  | 特性 | DFS 法 | Kahn 法 (BFS) |
  | --- | --- | --- |
  | 数据结构 | 栈/递归 | 队列 + 入度数组 |
  | 输出顺序 | 后序逆序 | 入度为0的顺序 |
  | 环检测 | onPath 标记 | 结果长度 < V |
  | 代码量 | 较少 | 较多 |
  | 字典序最小 | 不方便 | 配合优先队列 |

## 应用场景

  - **课程安排：**判断是否能修完所有课程（LeetCode 207/210）
  - **编译系统：**Makefile 中的依赖顺序
  - **任务调度：**有先后依赖的任务执行顺序
  - **数据流管道：**Spark/Flink 的算子执行顺序

  点击按钮查看结果
