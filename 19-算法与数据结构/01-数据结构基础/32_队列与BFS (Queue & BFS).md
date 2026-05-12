# Queue & BFS

### 队列在 BFS 中的角色

广度优先搜索（BFS）使用队列按层级逐层遍历图或树。将起点入队，每次取出队头元素，将其未访问的邻居入队，直到队列为空。保证了同层节点先于下层节点被访问。

### 关键特性

- **逐层遍历**：天然适合求最短路径（无权图）
- **判重机制**：配合 Set 或 visited 数组避免重复访问
- **队列为空即结束**：所有可达节点都已被访问

### 时间与空间复杂度

| 场景 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 树的层序遍历 | O(n) | O(w) w为最大宽度 |
| 图的 BFS | O(V+E) | O(V) |
| 最短路径（无权） | O(V+E) | O(V) |

### 适用场景 vs 替代方案

- **最短路径**：无权图 BFS 最优，有权图用 Dijkstra
- **层序遍历**：BFS 是标准方法，DFS 需额外记录深度
- **连通性判断**：BFS/DFS 均可，BFS 适合找最短路径
- **替代**：DFS 用栈实现，适合搜索所有路径或检测环

### 常见陷阱

- 忘记标记已访问节点，导致死循环（尤其在有环图中）
- 用数组 shift() 模拟队列导致 O(n) 出队，大数据量超时
- 层序遍历时混淆每层的边界，结果不是按层输出

```
// BFS 遍历树
function bfs(root) {
  if (!root) return [];
  const q = [root], res = [];
  while (q.length) {
    const node = q.shift();
    res.push(node.val);
    if (node.left) q.push(node.left);
    if (node.right) q.push(node.right);
  }
  return res;
}
const tree = {val:1, left:{val:2,left:null,right:null}, right:{val:3,left:{val:4,left:null,right:null},right:null}};
console.log(bfs(tree)); // [1,2,3,4]
```


### 实际应用

- **社交网络**：六度分隔理论，BFS 计算人际关系距离
- **地图导航**：无权图中找最短路线
- **搜索引擎爬虫**：从种子 URL 出发逐层抓取网页
- **游戏 AI**：迷宫寻路、八数码问题的状态空间搜索

  点击按钮查看结果
