# 图的遍历 - DFS (DFS Traversal)

  DFS 使用栈（递归或显式栈）尽可能深地探索图，当没有未访问的邻接顶点时回溯。

## 什么是 DFS

  深度优先搜索是一种沿一条路径尽可能深入，直到无法继续再回溯的遍历策略。DFS 适合解决需要穷举所有可能性的问题，如路径搜索、连通性判断、环检测等。递归实现利用调用栈，迭代实现使用显式栈。

## 关键性质

    - DFS 会生成一棵 DFS 搜索树（或森林），记录遍历的父子关系
    - 可以记录进入时间（tin）和离开时间（tout），判断祖先关系
    - 递归深度等于搜索树的最大深度，过深会导致栈溢出
    - 通过维护"当前路径"数组可以检测环

## 复杂度分析

    - **时间：**O(V+E)，每个顶点和每条边各访问一次
    - **空间：**O(V)，visited 集合 + 递归栈/显式栈

## 适用场景 vs BFS

    - 探索所有路径（如排列组合、迷宫所有走法）：DFS 更合适
    - 连通分量、强连通分量：DFS 是标准做法
    - 拓扑排序（DFS 后序逆序）：DFS 天然适用
    - 最短路径：应使用 BFS（无权图）或 Dijkstra（有权图）

## 常见陷阱

    - JavaScript 递归深度限制约为 10000 层，大图需改用迭代实现
    - 忘记在递归返回后恢复状态（如 onPath 标记）会导致环检测错误
    - 无向图 DFS 中回边（指向已访问父节点的边）不算环

```javascript
// 递归实现
function dfs(graph, v, visited = new Set()) {
  visited.add(v);
  console.log('访问:', v);
  for (const neighbor of graph[v]) {
    if (!visited.has(neighbor)) {
      dfs(graph, neighbor, visited);
    }
  }
}

// 迭代实现（避免栈溢出）
function dfsIterative(graph, start) {
  const visited = new Set();
  const stack = [start];
  while (stack.length) {
    const v = stack.pop();
    if (visited.has(v)) continue;
    visited.add(v);
    console.log('访问:', v);
    // 反向遍历使结果与递归一致
    for (let i = graph[v].length - 1; i >= 0; i--) {
      if (!visited.has(graph[v][i])) {
        stack.push(graph[v][i]);
      }
    }
  }
}
```

```cpp
// C++ 递归实现
void dfs(int u, vector<bool>& visited, vector<vector<int>>& graph) {
    visited[u] = true;
    cout << "访问: " << u << endl;
    for (int v : graph[u]) {
        if (!visited[v]) dfs(v, visited, graph);
    }
}
```

## 记录进入/离开时间

```javascript
function dfsWithTime(graph, n) {
  const visited = new Array(n).fill(false);
  const tin = new Array(n).fill(0);   // 进入时间
  const tout = new Array(n).fill(0);  // 离开时间
  let timer = 0;

  function dfs(u) {
    visited[u] = true;
    tin[u] = timer++;
    for (const v of graph[u]) {
      if (!visited[v]) dfs(v);
    }
    tout[u] = timer++;
  }

  for (let i = 0; i < n; i++) if (!visited[i]) dfs(i);
  return { tin, tout };
}
```

## 环检测

```javascript
// 有向图环检测
function hasCycleDirected(graph, n) {
  const visited = new Array(n).fill(false);
  const onPath = new Array(n).fill(false);

  function dfs(u) {
    visited[u] = true;
    onPath[u] = true;
    for (const v of (graph[u] || [])) {
      if (onPath[v]) return true;   // 找到环
      if (!visited[v] && dfs(v)) return true;
    }
    onPath[u] = false;  // 回溯时恢复
    return false;
  }

  for (let i = 0; i < n; i++) if (!visited[i]) if (dfs(i)) return true;
  return false;
}
```

## 实际应用

  在文件系统中递归遍历所有子目录和文件。在数独求解中，DFS 回溯法尝试每个空格填入数字，若冲突则回溯重试。
