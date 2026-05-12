# BFS 层序遍历应用 (BFS Applications)

  BFS 不仅可以遍历图，还有许多实际应用场景。这些应用的共同特点是：需要按距离/层次逐层处理，或找无权图的最短路径。

## 常见应用详解

    - **社交网络：**推荐好友、"你可能认识的人"——BFS 找距离为 2 的顶点

    - **网络爬虫：**广度优先抓取网页，优先处理距离首页近的页面

    - **GPS 导航：**无权图最短路径，每条道路权值相同

    - **二分图判定：**BFS 染色法，相邻顶点交替染色

    - **单词变换：**Word Ladder 问题，每次改变一个字母从单词 A 变到单词 B

    - **层序遍历：**二叉树按层输出，每层处理完毕再进入下一层

## 分层 BFS 技巧

  标准 BFS 不区分"层"，分层 BFS 通过在每轮循环开始时记录当前队列大小，确保一次处理一整层。这在需要按层统计结果（如二叉树层序遍历、多源 BFS 扩散）时非常有用。

## 复杂度分析

    - **时间：**O(V+E)，每个顶点和每条边各处理一次
    - **空间：**O(V)，队列和 visited 集合

## 关键性质

    - 分层 BFS 需要在 while 循环内用 for 循环处理当前层
    - 多源 BFS 将多个起点同时加入队列，求各顶点到最近起点的距离
    - BFS 可用于检测无向图是否为二分图（相邻顶点颜色不同）

## 常见陷阱

    - 分层 BFS 中忘记保存 `levelSize` 导致层统计出错
    - Word Ladder 中需要预处理邻居关系，不能每次都比较所有单词


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
}
```

## 交互演示：分层遍历
