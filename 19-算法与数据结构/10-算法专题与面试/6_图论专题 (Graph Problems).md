# 图论专题 (Graph Problems)

## 一、概念定义与原理

### 1.1 图的基础

**图 (Graph)** 由顶点集合 $V$ 和边集合 $E$ 组成，记为 $G = (V, E)$。

**分类：**
- **有向图 vs 无向图：** 边是否有方向
- **加权图 vs 无权图：** 边是否有权重
- **连通图 vs 非连通图：** 任意两点间是否有路径
- **有环图 vs 无环图 (DAG)：** 是否包含环

### 1.2 图的表示

**邻接矩阵：** `graph[u][v] = 1` 表示有边，适合稠密图，空间 $O(V^2)$

**邻接表：** 每个顶点维护一个邻居列表，适合稀疏图，空间 $O(V+E)$

```python
# 邻接表表示
graph = {
    0: [1, 2],
    1: [2],
    2: [0, 3],
    3: [3]
}
```

---

## 二、核心算法

### 2.1 BFS — 广度优先搜索

用于最短路径（无权图）、层序遍历。

```python
from collections import deque

def bfs(graph, start):
    visited = {start}
    queue = deque([start])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return order
```

### 2.2 DFS — 深度优先搜索

用于连通分量、拓扑排序、环检测。

```python
def dfs(graph, start):
    visited = set()
    order = []

    def _dfs(node):
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                _dfs(neighbor)

    _dfs(start)
    return order
```

### 2.3 拓扑排序 (Kahn算法)

```python
from collections import deque

def topo_sort(num_nodes, edges):
    graph = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque(i for i in range(num_nodes) if in_degree[i] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_nodes else []  # 空表示有环
```

### 2.4 并查集 (Union-Find)

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py: return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
```

### 2.5 Dijkstra 最短路径

```python
import heapq

def dijkstra(graph, start):
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]: continue
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dist
```

---

## 三、经典题目详解

### 3.1 岛屿数量 (LeetCode 200)

```python
def num_islands(grid):
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        dfs(i+1, j); dfs(i-1, j)
        dfs(i, j+1); dfs(i, j-1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count
```

### 3.2 课程表 (LeetCode 207) — 拓扑排序

```python
def can_finish(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses
    for a, b in prerequisites:
        graph[b].append(a)
        in_degree[a] += 1

    queue = [i for i in range(num_courses) if in_degree[i] == 0]
    count = 0
    while queue:
        node = queue.pop()
        count += 1
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    return count == num_courses
```

### 3.3 冗余连接 (LeetCode 684) — 并查集

```python
def find_redundant_connection(edges):
    uf = UnionFind(len(edges) + 1)
    for u, v in edges:
        if not uf.union(u, v):
            return [u, v]
```

### 3.4 二分图判定 (LeetCode 785)

```python
def is_bipartite(graph):
    color = {}
    for node in range(len(graph)):
        if node not in color:
            color[node] = 0
            stack = [node]
            while stack:
                curr = stack.pop()
                for neighbor in graph[curr]:
                    if neighbor not in color:
                        color[neighbor] = 1 - color[curr]
                        stack.append(neighbor)
                    elif color[neighbor] == color[curr]:
                        return False
    return True
```

---

## 四、C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

// 并查集
class UnionFind {
    vector<int> parent, rank_;
public:
    UnionFind(int n) : parent(n), rank_(n, 0) {
        iota(parent.begin(), parent.end(), 0);
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank_[px] < rank_[py]) swap(px, py);
        parent[py] = px;
        if (rank_[px] == rank_[py]) rank_[px]++;
        return true;
    }
};

// Dijkstra
vector<int> dijkstra(vector<vector<pair<int,int>>>& graph, int start) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    dist[start] = 0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

---

## 五、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| BFS/DFS | $O(V+E)$ | $O(V)$ |
| 拓扑排序 | $O(V+E)$ | $O(V)$ |
| Dijkstra(堆) | $O((V+E)\log V)$ | $O(V)$ |
| Bellman-Ford | $O(VE)$ | $O(V)$ |
| Floyd-Warshall | $O(V^3)$ | $O(V^2)$ |
| Kruskal MST | $O(E \log E)$ | $O(V)$ |
| 并查集(路径压缩) | $O(\alpha(n))$ 操作 | $O(n)$ |

---

## 六、面试高频题

1. **LeetCode 200：** 岛屿数量
2. **LeetCode 207：** 课程表（拓扑排序）
3. **LeetCode 684：** 冗余连接
4. **LeetCode 785：** 二分图
5. **LeetCode 133：** 克隆图
6. **LeetCode 994：** 腐烂的橘子（多源BFS）
7. **LeetCode 127：** 单词接龙
8. **LeetCode 743：** 网络延迟时间
9. **LeetCode 329：** 矩阵中的最长递增路径
10. **LeetCode 399：** 除法求值（并查集）
