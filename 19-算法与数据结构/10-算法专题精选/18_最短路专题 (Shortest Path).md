# 最短路专题 (Shortest Path)

## 一、概念定义与原理

### 1.1 最短路问题分类

| 类型 | 算法 | 复杂度 | 适用条件 |
|------|------|--------|---------|
| 单源最短路 | Dijkstra | $O((V+E)\log V)$ | 非负权 |
| 单源最短路 | Bellman-Ford | $O(VE)$ | 可有负权 |
| 单源最短路 | SPFA | $O(VE)$ 最坏 | 可有负权 |
| 全源最短路 | Floyd | $O(V^3)$ | 任意 |

---

## 二、核心算法

### 2.1 Dijkstra

贪心思想，每次取距离最小的未访问节点，更新其邻居。用优先队列优化。

### 2.2 Bellman-Ford

对所有边进行 $V-1$ 轮松弛。第 $k$ 轮后，保证最多经过 $k$ 条边的最短路正确。

第 $V$ 轮仍可松弛说明存在负环。

### 2.3 SPFA（队列优化 Bellman-Ford）

用队列维护需要松弛的节点。平均效率优于 Bellman-Ford，但最坏仍是 $O(VE)$。

### 2.4 Floyd

三重循环枚举中转点 $k$：`dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])`

---

## 三、代码实现

### 3.1 Dijkstra - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<long long> dijkstra(vector<vector<pair<int,long long>>>& g, int s) {
    int n = g.size();
    vector<long long> dist(n, LLONG_MAX);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    dist[s] = 0; pq.push({0, s});
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;
        for (auto [v, w] : g[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

### 3.2 Bellman-Ford - C++

```cpp
// 返回是否有负环
bool bellman_ford(vector<tuple<int,int,long long>>& edges, int n, int s,
                  vector<long long>& dist) {
    dist.assign(n, LLONG_MAX);
    dist[s] = 0;
    for (int i = 0; i < n - 1; i++) {
        for (auto [u, v, w] : edges) {
            if (dist[u] != LLONG_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    // 检测负环
    for (auto [u, v, w] : edges) {
        if (dist[u] != LLONG_MAX && dist[u] + w < dist[v]) return true;
    }
    return false;
}
```

### 3.3 SPFA - C++

```cpp
bool spfa(vector<vector<pair<int,long long>>>& g, int s,
          vector<long long>& dist) {
    int n = g.size();
    dist.assign(n, LLONG_MAX);
    vector<bool> inq(n, false);
    vector<int> cnt(n, 0);
    queue<int> q;
    dist[s] = 0; inq[s] = true; q.push(s);
    while (!q.empty()) {
        int u = q.front(); q.pop(); inq[u] = false;
        for (auto [v, w] : g[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!inq[v]) {
                    q.push(v); inq[v] = true;
                    if (++cnt[v] >= n) return true; // 负环
                }
            }
        }
    }
    return false;
}
```

### 3.4 Floyd - C++

```cpp
void floyd(vector<vector<long long>>& dist, int n) {
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (dist[i][k] < LLONG_MAX && dist[k][j] < LLONG_MAX)
                    dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
}
```

### 3.5 Python 实现

```python
import heapq

def dijkstra(graph, s):
    dist = {s: 0}; pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float('inf')): continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist.get(v, float('inf')):
                dist[v] = nd; heapq.heappush(pq, (nd, v))
    return dist

def floyd(dist, n):
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

# 测试
INF = float('inf')
graph = {0: [(1,4),(2,1)], 1: [(3,1)], 2: [(1,2),(3,5)], 3: []}
print(dijkstra(graph, 0))  # {0:0, 2:1, 1:3, 3:4}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 | 负权 |
|------|-----------|-----------|------|
| Dijkstra | $O((V+E)\log V)$ | $O(V)$ | 不支持 |
| Bellman-Ford | $O(VE)$ | $O(V)$ | 支持 |
| SPFA | $O(VE)$ 最坏 | $O(V)$ | 支持 |
| Floyd | $O(V^3)$ | $O(V^2)$ | 支持 |

---

## 五、竞赛与面试应用场景

1. **LeetCode 743：** 网络延迟时间（Dijkstra）
2. **LeetCode 787：** K站中转最便宜航班（Bellman-Ford）
3. **LeetCode 1334：** 阈值距离内邻居最少的城市
4. **负权检测：** Bellman-Ford / SPFA
5. **全源最短路：** Floyd
