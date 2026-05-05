## 深度优先搜索 (DFS)

  DFS 使用栈（递归或显式栈）尽可能深地探索图，当没有未访问的邻接顶点时回溯。


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
}```

  ## 算法特点


    - 适合探索所有可能路径、连通分量

    - 递归实现简洁，但深度过大可能栈溢出

    - 可用显式栈实现迭代版本

    - 时间复杂度 O(V+E)



  ## 交互演示
