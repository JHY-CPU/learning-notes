# 44-图论面试题精讲 (Graph Interview Problems)

图论面试高频题分类与解题思路。

## 高频题型分类

| 类型 | 代表题目 | 核心技巧 |
|------|---------|---------|
| 图的遍历 | 克隆图、岛屿数量 | BFS/DFS |
| 拓扑排序 | 课程表、任务调度 | 入度 + BFS |
| 最短路径 | 网络延迟时间、单词接龙 | Dijkstra/BFS |
| 连通性 | 省份数量、冗余连接 | DFS/并查集 |
| 环检测 | 课程表（有向环）、冗余连接（无向环） | DFS着色/并查集 |
| 二分图 | 可能的二分法 | BFS着色/DFS |
| 最小生成树 | 连接所有城市的最小费用 | Kruskal/Prim |
| 并查集 | 等式方程可满足性 | Union-Find |

## JavaScript 实现

```javascript
// 1. 岛屿数量 (LeetCode 200) - DFS
function numIslands(grid) {
  let count = 0;
  function dfs(i, j) {
    if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] !== '1') return;
    grid[i][j] = '0';
    dfs(i + 1, j); dfs(i - 1, j); dfs(i, j + 1); dfs(i, j - 1);
  }
  for (let i = 0; i < grid.length; i++)
    for (let j = 0; j < grid[0].length; j++)
      if (grid[i][j] === '1') { count++; dfs(i, j); }
  return count;
}

// 2. 课程表 - 拓扑排序 (LeetCode 207)
function canFinish(numCourses, prerequisites) {
  const graph = Array.from({ length: numCourses }, () => []);
  const inDegree = new Array(numCourses).fill(0);
  for (const [a, b] of prerequisites) {
    graph[b].push(a);
    inDegree[a]++;
  }
  const queue = [];
  for (let i = 0; i < numCourses; i++) if (inDegree[i] === 0) queue.push(i);
  let count = 0;
  while (queue.length) {
    const u = queue.shift();
    count++;
    for (const v of graph[u]) if (--inDegree[v] === 0) queue.push(v);
  }
  return count === numCourses;
}

// 3. 省份数量 - 并查集 (LeetCode 547)
function findCircleNum(isConnected) {
  const n = isConnected.length;
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x) { while (parent[x] !== x) x = parent[x] = parent[parent[x]]; return x; }
  function union(a, b) { parent[find(a)] = find(b); }
  for (let i = 0; i < n; i++)
    for (let j = i + 1; j < n; j++)
      if (isConnected[i][j]) union(i, j);
  let count = 0;
  for (let i = 0; i < n; i++) if (parent[i] === i) count++;
  return count;
}

// 4. 二分图判定 (LeetCode 785) - BFS着色
function isBipartite(graph) {
  const n = graph.length;
  const color = new Array(n).fill(-1);
  for (let i = 0; i < n; i++) {
    if (color[i] !== -1) continue;
    color[i] = 0;
    const queue = [i];
    while (queue.length) {
      const u = queue.shift();
      for (const v of graph[u]) {
        if (color[v] === -1) { color[v] = 1 - color[u]; queue.push(v); }
        else if (color[v] === color[u]) return false;
      }
    }
  }
  return true;
}

// 5. 克隆图 (LeetCode 133)
function cloneGraph(node) {
  if (!node) return null;
  const map = new Map();
  function clone(node) {
    if (map.has(node)) return map.get(node);
    const copy = { val: node.val, neighbors: [] };
    map.set(node, copy);
    for (const n of node.neighbors) copy.neighbors.push(clone(n));
    return copy;
  }
  return clone(node);
}

// 测试
console.log(numIslands([
  ['1','1','0','0','0'],
  ['1','1','0','0','0'],
  ['0','0','1','0','0'],
  ['0','0','0','1','1']
])); // 3
console.log(canFinish(4, [[1,0],[2,0],[3,1],[3,2]])); // true
```

## C++ 实现

```cpp
#include <vector>
#include <queue>
using namespace std;

// 岛屿数量
int numIslands(vector<vector<char>>& grid) {
    int m = grid.size(), n = grid[0].size(), count = 0;
    function<void(int,int)> dfs = [&](int i, int j) {
        if (i < 0 || i >= m || j < 0 || j >= n || grid[i][j] != '1') return;
        grid[i][j] = '0';
        dfs(i+1,j); dfs(i-1,j); dfs(i,j+1); dfs(i,j-1);
    };
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (grid[i][j] == '1') { count++; dfs(i,j); }
    return count;
}

// 拓扑排序
bool canFinish(int n, vector<vector<int>>& prereq) {
    vector<vector<int>> graph(n);
    vector<int> indeg(n, 0);
    for (auto& p : prereq) { graph[p[1]].push_back(p[0]); indeg[p[0]]++; }
    queue<int> q;
    for (int i = 0; i < n; i++) if (indeg[i] == 0) q.push(i);
    int cnt = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop(); cnt++;
        for (int v : graph[u]) if (--indeg[v] == 0) q.push(v);
    }
    return cnt == n;
}
```

## 解题套路

1. **拿到图论题先判断类型**：遍历？最短路？连通性？匹配？
2. **选数据结构**：邻接表 vs 邻接矩阵（稀疏图用邻接表）
3. **选算法**：BFS 求最短路，DFS 求路径/回溯，并查集求连通
4. **注意边界**：空图、单节点、不连通图

## 常见陷阱

1. **忘判不连通**：图可能有多个连通分量
2. **有向 vs 无向**：环检测在有向和无向图中方法不同
3. **图中有环**：BFS/DFS 需要 visited 避免死循环
4. **自环和重边**：某些问题需要特殊处理
