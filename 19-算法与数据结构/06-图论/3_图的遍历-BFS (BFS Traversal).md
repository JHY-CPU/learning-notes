# 图的遍历 - BFS (BFS Traversal)

  BFS 使用队列（Queue）逐层遍历图。先访问距离起点最近的顶点，再逐层向外扩展。

## 什么是 BFS

  广度优先搜索是一种逐层遍历图的算法。从起始顶点出发，先访问所有距离为 1 的顶点，再访问距离为 2 的顶点，以此类推。BFS 保证了按距离递增的顺序访问顶点，这使其成为无权图最短路径问题的天然解法。

## 关键性质

    - BFS 天然按照顶点到起点的距离（边数）分层访问
    - 使用队列维护待访问顶点，保证先进先出的顺序
    - 每个顶点最多入队一次，确保不会重复访问
    - BFS 生成的搜索树中，从根到任意节点的路径是最短路径

## 复杂度分析

    - **时间：**O(V+E)，每个顶点和每条边各访问一次
    - **空间：**O(V)，visited 集合和队列最多存储 V 个顶点

## 适用场景 vs DFS

    - 求无权图最短路径：BFS 优于 DFS
    - 层序遍历（如二叉树按层输出）：BFS 是标准做法
    - 从近到远搜索（如社交网络中找距离最近的目标人）
    - DFS 更适合路径探索、回溯、连通分量等场景

## 常见陷阱

    - JavaScript 中 `Array.shift()` 是 O(n) 操作，大量数据应用双端队列
    - 忘记在入队时标记已访问，会导致同一顶点重复入队
    - 图不连通时，需要对每个未访问的顶点启动一次 BFS

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
}
```

## 实际应用

  在迷宫求解中，BFS 可以找到从起点到终点的最短路径（最少步数）。在网络爬虫中，BFS 实现从首页出发逐层抓取链接，确保优先抓取距离首页较近的页面。

## 算法步骤


    - 将起始顶点标记为已访问并入队

    - 从队列取出一个顶点，访问它

    - 将其所有未访问的邻接顶点标记并入队

    - 重复步骤 2-3 直到队列为空
