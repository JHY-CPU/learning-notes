# 图论专题 (Graph Problems)

## 一、概念定义与原理

### 1.1 图的表示

- **邻接矩阵：** `g[i][j]` 表示边权，$O(V^2)$ 空间
- **邻接表：** 每个节点维护一个邻居列表，$O(V+E)$ 空间
- **边列表：** 存储所有边，适合 Kruskal 算法

### 1.2 图的分类

- **有向图 / 无向图**
- **带权图 / 无权图**
- **稀疏图（$E \ll V^2$）/ 稠密图（$E \approx V^2$）**

---

## 二、核心算法

### 2.1 BFS（广度优先搜索）

用途：无权图最短路、层级遍历、连通分量。

### 2.2 DFS（深度优先搜索）

用途：连通分量、拓扑排序、环检测、强连通分量。

### 2.3 最短路

- **Dijkstra：** 非负权图，$O((V+E) \log V)$
- **Bellman-Ford：** 可处理负权，$O(VE)$
- **Floyd：** 全源最短路，$O(V^3)$

### 2.4 拓扑排序

DAG（有向无环图）的线性排序。Kahn 算法（BFS）或 DFS。

---

## 三、代码实现

### 3.1 BFS 最短路 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> bfs_shortest(vector<vector<int>>& graph, int start) {
    int n = graph.size();
    vector<int> dist(n, -1);
    queue<int> q;
    dist[start] = 0;
    q.push(start);
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : graph[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}
```

### 3.2 Dijkstra - C++

```cpp
vector<long long> dijkstra(vector<vector<pair<int,long long>>>& graph, int start) {
    int n = graph.size();
    vector<long long> dist(n, LLONG_MAX);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[start] = 0;
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

### 3.3 拓扑排序（Kahn）- C++

```cpp
vector<int> topological_sort(vector<vector<int>>& graph, vector<int>& indegree) {
    int n = graph.size();
    queue<int> q;
    for (int i = 0; i < n; i++) {
        if (indegree[i] == 0) q.push(i);
    }
    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : graph[u]) {
            if (--indegree[v] == 0) q.push(v);
        }
    }
    return order.size() == n ? order : vector<int>(); // 有环返回空
}
```

### 3.4 Python 实现

```python
from collections import deque, defaultdict
import heapq

def bfs(graph, start):
    dist = {start: 0}; q = deque([start])
    while q:
        u = q.popleft()
        for v in graph[u]:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist

def dijkstra(graph, start):
    dist = {start: 0}; pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')): continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist

def topological_sort(graph, n):
    indegree = [0] * n
    for u in range(n):
        for v in graph[u]: indegree[v] += 1
    q = deque(i for i in range(n) if indegree[i] == 0)
    order = []
    while q:
        u = q.popleft(); order.append(u)
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0: q.append(v)
    return order if len(order) == n else []
```

### 3.5 Floyd 全源最短路

```cpp
void floyd(vector<vector<long long>>& dist, int n) {
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (dist[i][k] < LLONG_MAX && dist[k][j] < LLONG_MAX)
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 | 适用条件 |
|------|-----------|-----------|---------|
| BFS | $O(V+E)$ | $O(V)$ | 无权图 |
| DFS | $O(V+E)$ | $O(V)$ | 通用 |
| Dijkstra | $O((V+E)\log V)$ | $O(V)$ | 非负权 |
| Bellman-Ford | $O(VE)$ | $O(V)$ | 可有负权 |
| Floyd | $O(V^3)$ | $O(V^2)$ | 全源最短路 |
| 拓扑排序 | $O(V+E)$ | $O(V)$ | DAG |

---

## 五、竞赛与面试应用场景

1. **LeetCode 207：** 课程表（拓扑排序）
2. **LeetCode 743：** 网络延迟时间（Dijkstra）
3. **LeetCode 133：** 克隆图（BFS/DFS）
4. **LeetCode 210：** 课程表II
5. **连通分量/环检测：** DFS 经典应用
