# 拓扑排序应用 (Topological Applications)

  拓扑排序是解决有依赖关系的线性排序问题的核心工具。以下是最常见的应用场景。

## 常见应用

    - **课程安排：**大学课程先修关系，确定修课顺序（LeetCode 207/210）

    - **编译依赖：**Makefile / 包管理器按依赖顺序编译和安装

    - **任务调度：**有依赖关系的任务（如数据流水线）确定执行顺序

    - **数据处理：**DAG 中按依赖顺序执行计算（如 TensorFlow 计算图）

    - **项目管理：**关键路径法 (CPM) 中的任务排序和工期计算

## 关键性质

    - 拓扑排序的入度为 0 的顶点是"无依赖"的任务，可立即执行
    - 如果排序结果少于 V 个顶点，说明存在循环依赖
    - 关键路径 = 最长路径，决定项目最短完成时间

## 复杂度分析

    - **时间：**O(V+E)
    - **空间：**O(V)

## 常见陷阱

    - 课程安排中，需要先判断是否有环（循环依赖），再输出排序
    - 关键路径计算需要在拓扑排序基础上做 DP 求最长路
    - 并行执行时，同一批入度为 0 的任务可以同时执行



## LeetCode 相关题目


    - 207. 课程表 (Course Schedule)

    - 210. 课程表 II (Course Schedule II)

    - 269. 火星词典 (Alien Dictionary)

    - 329. 矩阵中的最长递增路径



## 关键路径法（CPM）

```javascript
// 关键路径：求项目最短完成时间
function criticalPath(adj, durations, n) {
  // 先做拓扑排序
  const inDegree = new Array(n).fill(0);
  for (const u in adj) for (const v of adj[u]) inDegree[v]++;
  const queue = [];
  for (let i = 0; i < n; i++) if (inDegree[i] === 0) queue.push(i);

  const topo = [];
  const ve = new Array(n).fill(0);  // 最早开始时间
  while (queue.length) {
    const u = queue.shift();
    topo.push(u);
    for (const v of (adj[u] || [])) {
      ve[v] = Math.max(ve[v], ve[u] + durations[u]);
      if (--inDegree[v] === 0) queue.push(v);
    }
  }

  // 最晚开始时间（反向）
  const vl = new Array(n).fill(Infinity);
  vl[topo[n-1]] = ve[topo[n-1]];
  for (let i = n - 2; i >= 0; i--) {
    const u = topo[i];
    for (const v of (adj[u] || [])) {
      vl[u] = Math.min(vl[u], vl[v] - durations[u]);
    }
  }

  // 关键活动：最早=最晚
  const critical = [];
  for (let i = 0; i < n; i++) {
    if (Math.abs(ve[i] - vl[i]) < 1e-9) critical.push(i);
  }
  return { projectDuration: Math.max(...ve), critical };
}
```

## 代码示例：课程安排

```javascript
// LeetCode 210: 课程表 II
function findOrder(numCourses, prerequisites) {
  const graph = Array.from({length: numCourses}, () => []);
  const inDegree = new Array(numCourses).fill(0);

  for (const [course, pre] of prerequisites) {
    graph[pre].push(course);
    inDegree[course]++;
  }

  const queue = [];
  for (let i = 0; i < numCourses; i++) {
    if (inDegree[i] === 0) queue.push(i);
  }

  const order = [];
  while (queue.length) {
    const u = queue.shift();
    order.push(u);
    for (const v of graph[u]) {
      if (--inDegree[v] === 0) queue.push(v);
    }
  }

  return order.length === numCourses ? order : [];
}
console.log(findOrder(4, [[1,0],[2,0],[3,1],[3,2]]));
// [0, 1, 2, 3] 或 [0, 2, 1, 3]
```

## 交互演示：课程安排

  课程依赖: C1→(C3,C4), C2→C4, C3→C5, C4→(C5,C6)
