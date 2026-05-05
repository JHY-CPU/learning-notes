## BFS 应用场景


    - **社交网络：**推荐好友、"你可能认识的人"

    - **网络爬虫：**广度优先抓取网页

    - **GPS 导航：**无权图最短路径

    - **二分图判定：**染色法检测

    - **单词变换：**Word Ladder 问题

    - **层序遍历：**按层级处理节点



  ## 分层 BFS


```javascript
function levelOrder(graph, start) {
  const queue = [start], visited = new Set([start]);
  const levels = [];
  while (queue.length) {
    const levelSize = queue.length;
    const currentLevel = [];
    for (let i = 0; i < levelSize; i++) {
      const v = queue.shift();
      currentLevel.push(v);
      for (const w of graph[v])
        if (!visited.has(w)) { visited.add(w); queue.push(w); }
    }
    levels.push(currentLevel);
  }
  return levels;
}```

  ## 交互演示：分层遍历
