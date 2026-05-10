# 搜索专题 (Search Problems)

## 一、概念定义与原理

### 1.1 搜索算法分类

| 算法 | 特点 | 适用场景 |
|------|------|---------|
| BFS | 逐层扩展，找最短路 | 无权图最短路、层次搜索 |
| DFS | 深入到底，回溯 | 路径枚举、连通性 |
| 迭代加深DFS | 限制深度的DFS | 深度有限、避免无限搜索 |
| A* | 启发式搜索 | 最优路径、游戏AI |
| 双向BFS | 两端同时搜索 | 起点终点都已知 |

### 1.2 剪枝策略

- **可行性剪枝：** 当前状态已不满足约束
- **最优性剪枝：** 当前代价已超过最优解
- **记忆化：** 避免重复搜索相同状态
- **对称性剪枝：** 跳过等价状态

---

## 二、核心算法

### 2.1 BFS 模板

用队列维护待扩展状态，visited 数组去重。

### 2.2 DFS 模板

递归或栈实现，visited 标记已访问状态。

### 2.3 A* 算法

$f(n) = g(n) + h(n)$，其中 $g(n)$ 是起点到 $n$ 的实际代价，$h(n)$ 是 $n$ 到终点的启发式估计。$h$ 需满足**可采纳性**（不高估）。

### 2.4 迭代加深

从深度限制 1 开始，逐步增加深度限制，直到找到解。

---

## 三、代码实现

### 3.1 BFS 求最短路径 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 迷宫最短路
int bfs_maze(vector<vector<int>>& maze, pair<int,int> start, pair<int,int> end) {
    int m = maze.size(), n = maze[0].size();
    vector<vector<bool>> visited(m, vector<bool>(n, false));
    queue<pair<int,int>> q;
    q.push(start); visited[start.first][start.second] = true;
    int steps = 0;
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    while (!q.empty()) {
        int size = q.size();
        for (int i = 0; i < size; i++) {
            auto [x, y] = q.front(); q.pop();
            if (x == end.first && y == end.second) return steps;
            for (int d = 0; d < 4; d++) {
                int nx = x + dx[d], ny = y + dy[d];
                if (nx >= 0 && nx < m && ny >= 0 && ny < n &&
                    !visited[nx][ny] && maze[nx][ny] == 0) {
                    visited[nx][ny] = true;
                    q.push({nx, ny});
                }
            }
        }
        steps++;
    }
    return -1;
}
```

### 3.2 A* 算法 - C++

```cpp
struct Node {
    int x, y, g, h;
    bool operator>(const Node& o) const { return g + h > o.g + o.h; }
};

int heuristic(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2); // 曼哈顿距离
}

int astar(vector<vector<int>>& grid, pair<int,int> start, pair<int,int> end) {
    int m = grid.size(), n = grid[0].size();
    priority_queue<Node, vector<Node>, greater<Node>> pq;
    vector<vector<int>> dist(m, vector<int>(n, INT_MAX));
    dist[start.first][start.second] = 0;
    pq.push({start.first, start.second, 0, heuristic(start.first, start.second, end.first, end.second)});
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    while (!pq.empty()) {
        auto [x, y, g, h] = pq.top(); pq.pop();
        if (x == end.first && y == end.second) return g;
        if (g > dist[x][y]) continue;
        for (int d = 0; d < 4; d++) {
            int nx = x+dx[d], ny = y+dy[d];
            if (nx>=0 && nx<m && ny>=0 && ny<n && grid[nx][ny]==0) {
                int ng = g + 1;
                if (ng < dist[nx][ny]) {
                    dist[nx][ny] = ng;
                    pq.push({nx, ny, ng, heuristic(nx, ny, end.first, end.second)});
                }
            }
        }
    }
    return -1;
}
```

### 3.3 Python 实现

```python
from collections import deque
import heapq

def bfs_grid(grid, start, end):
    m, n = len(grid), len(grid[0])
    visited = set([start]); q = deque([(start, 0)])
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    while q:
        (x,y), d = q.popleft()
        if (x,y) == end: return d
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0<=nx<m and 0<=ny<n and (nx,ny) not in visited and grid[nx][ny]==0:
                visited.add((nx,ny)); q.append(((nx,ny), d+1))
    return -1

def astar(grid, start, end):
    def h(p): return abs(p[0]-end[0]) + abs(p[1]-end[1])
    m, n = len(grid), len(grid[0])
    pq = [(h(start), 0, start)]; dist = {start: 0}
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    while pq:
        f, g, (x,y) = heapq.heappop(pq)
        if (x,y) == end: return g
        if g > dist.get((x,y), float('inf')): continue
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0<=nx<m and 0<=ny<n and grid[nx][ny]==0:
                ng = g + 1
                if ng < dist.get((nx,ny), float('inf')):
                    dist[(nx,ny)] = ng
                    heapq.heappush(pq, (ng+h((nx,ny)), ng, (nx,ny)))
    return -1

# 测试
grid = [[0,0,1,0],[0,1,0,0],[0,0,0,1],[1,0,0,0]]
print(bfs_grid(grid, (0,0), (3,3)))  # 6
```

### 3.4 双向BFS

```cpp
int bidirectional_bfs(vector<vector<int>>& grid, pair<int,int> start, pair<int,int> end) {
    if (start == end) return 0;
    int m = grid.size(), n = grid[0].size();
    unordered_map<long long, int> dist_start, dist_end;
    queue<pair<int,int>> q_start, q_end;
    q_start.push(start); dist_start[start.first*n+start.second] = 0;
    q_end.push(end); dist_end[end.first*n+end.second] = 0;
    int dx[] = {0,0,1,-1}, dy[] = {1,-1,0,0};
    while (!q_start.empty() && !q_end.empty()) {
        // 扩展 start 侧
        if (expand(q_start, dist_start, dist_end, grid, m, n, dx, dy))
            return dist_start[q_start.front().first*n+q_start.front().second]
                 + dist_end[q_start.front().first*n+q_start.front().second];
        // 扩展 end 侧
        if (expand(q_end, dist_end, dist_start, grid, m, n, dx, dy))
            return dist_end[q_end.front().first*n+q_end.front().second]
                 + dist_start[q_end.front().first*n+q_end.front().second];
    }
    return -1;
}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| BFS/DFS | $O(V+E)$ | $O(V)$ |
| A* | $O(b^d)$ 最坏 | $O(b^d)$ |
| 双向BFS | $O(b^{d/2})$ | $O(b^{d/2})$ |
| 迭代加深 | $O(b^d)$ | $O(d)$ |

$b$ 为分支因子，$d$ 为解的深度。

---

## 五、竞赛与面试应用场景

1. **LeetCode 127：** 单词接龙（BFS）
2. **LeetCode 1091：** 二进制矩阵中的最短路径
3. **LeetCode 773：** 滑动谜题（BFS）
4. **LeetCode 815：** 公交路线（BFS）
5. **游戏寻路：** A* 算法
