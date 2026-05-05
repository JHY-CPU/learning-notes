## 广度优先搜索 (BFS)

  BFS 使用队列（Queue）逐层遍历图。先访问距离起点最近的顶点，再逐层向外扩展。


```javascript
function bfs(graph, start) {
  const visited = new Set();
  const queue = [start];
  visited.add(start);
  while (queue.length > 0) {
    const vertex = queue.shift();
    console.log('访问:', vertex);
    for (const neighbor of graph[vertex]) {
      if (!visited.has(neighbor)) {
        visited.add(neighbor);
        queue.push(neighbor);
      }
    }
  }
}```

  ## 算法步骤


    - 将起始顶点标记为已访问并入队

    - 从队列取出一个顶点，访问它

    - 将其所有未访问的邻接顶点标记并入队

    - 重复步骤 2-3 直到队列为空



  ## 交互演示
