# Dijkstra算法实践 (Dijkstra Practice)

## 一、实际应用场景

### 1.1 典型应用

| 应用 | 说明 |
|------|------|
| 导航系统 | Google Maps、高德地图 |
| 网络路由 | OSPF协议中的最短路径 |
| 社交网络 | 最短连接路径 |
| 游戏开发 | NPC寻路算法 |
| 物流配送 | 最优配送路线 |

### 1.2 注意事项

> **Dijkstra 不能处理负权边！** 如果图中存在负权边，应使用 Bellman-Ford 或 SPFA 算法。

---

## 二、堆优化Dijkstra

### 2.1 算法步骤

1. 初始化：`dist[start] = 0`，其余为无穷大
2. 将起点加入最小堆
3. 取出距离最小的未访问节点 `u`
4. 更新 `u` 的所有邻居 `v`：`dist[v] = min(dist[v], dist[u] + w)`
5. 重复3-4直到堆为空

### 2.2 Python 实现

```python
import heapq

def dijkstra(graph, start, end=None):
    """
    graph: 邻接表 {node: [(neighbor, weight), ...]}
    返回: dist字典 或 到end的最短距离
    """
    dist = {node: float('inf') for node in graph}
    dist[start] = 0
    prev = {start: None}
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # 跳过已处理的节点

        if end is not None and u == end:
            return dist[end], reconstruct_path(prev, end)

        for v, w in graph[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist

def reconstruct_path(prev, end):
    path = []
    curr = end
    while curr is not None:
        path.append(curr)
        curr = prev[curr]
    return path[::-1]
```

### 2.3 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

vector<int> dijkstra(vector<vector<pair<int,int>>>& graph, int start) {
    int n = graph.size();
    vector<int> dist(n, INT_MAX);
    vector<int> prev(n, -1);
    dist[start] = 0;

    // 最小堆: (距离, 节点)
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0, start});

    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;

        for (auto [v, w] : graph[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                prev[v] = u;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

---

## 三、完整实践示例

### 3.1 城市导航

```python
def city_navigation():
    # 城市间距离图
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('C', 1), ('D', 5)],
        'C': [('D', 8), ('E', 10)],
        'D': [('E', 2)],
        'E': [('A', 7), ('D', 6)],
    }

    dist, path = dijkstra(graph, 'A', 'E')
    print(f"最短距离: {dist}")
    print(f"路径: {' -> '.join(path)}")

city_navigation()
# 最短距离: 10
# 路径: A -> C -> B -> D -> E
```

### 3.2 网格图Dijkstra

```python
def grid_dijkstra(grid, start, end):
    """网格图中的最短路径，每个格子有通行代价"""
    m, n = len(grid), len(grid[0])
    dist = [[float('inf')] * n for _ in range(m)]
    dist[start[0]][start[1]] = grid[start[0]][start[1]]
    heap = [(dist[start[0]][start[1]], start[0], start[1])]

    while heap:
        d, x, y = heapq.heappop(heap)
        if (x, y) == end:
            return d
        if d > dist[x][y]:
            continue
        for dx, dy in [(0,1),(0,-1),(1,0),(-1,0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < m and 0 <= ny < n:
                nd = d + grid[nx][ny]
                if nd < dist[nx][ny]:
                    dist[nx][ny] = nd
                    heapq.heappush(heap, (nd, nx, ny))

    return -1
```

---

## 四、变种问题

### 4.1 第K短路 (A*算法)

```python
def kth_shortest_path(graph, start, end, k):
    """使用A*算法找第K短路"""
    # 启发式函数：到终点的直线距离（或0=Dijkstra）
    def heuristic(node):
        return 0  # 退化为Dijkstra

    # 最小堆: (f = g + h, g, node, path)
    heap = [(heuristic(start), 0, start, [start])]
    count = {start: 0}

    while heap:
        f, g, u, path = heapq.heappop(heap)
        if u == end:
            count[u] = count.get(u, 0) + 1
            if count[u] == k:
                return g, path
            continue

        for v, w in graph.get(u, []):
            new_g = g + w
            heapq.heappush(heap, (new_g + heuristic(v), new_g, v, path + [v]))

    return -1, []
```

### 4.2 带约束的最短路径

```python
def constrained_dijkstra(graph, start, end, max_cost):
    """在总花费不超过max_cost的前提下找最短距离"""
    # state: (node, cost_used) → min_distance
    dist = {(start, 0): 0}
    heap = [(0, start, 0)]  # (distance, node, cost_used)

    while heap:
        d, u, c = heapq.heappop(heap)
        if u == end:
            return d
        if d > dist.get((u, c), float('inf')):
            continue
        for v, w, cost in graph.get(u, []):
            new_cost = c + cost
            if new_cost > max_cost:
                continue
            new_dist = d + w
            state = (v, new_cost)
            if new_dist < dist.get(state, float('inf')):
                dist[state] = new_dist
                heapq.heappush(heap, (new_dist, v, new_cost))

    return -1
```

---

## 五、性能优化

### 5.1 提前终止

当目标节点被访问时，可以提前终止（因为Dijkstra保证首次访问时即是最短距离）。

### 5.2 双向Dijkstra

从起点和终点同时搜索，相遇时停止。

### 5.3 分层图

对大规模图使用分层策略，减少搜索空间。

---

## 六、复杂度分析

| 实现 | 时间 | 空间 |
|------|------|------|
| 邻接矩阵 | $O(V^2)$ | $O(V^2)$ |
| 邻接表+堆 | $O((V+E)\log V)$ | $O(V+E)$ |
| Fibonacci堆 | $O(V \log V + E)$ | $O(V+E)$ |

---

## 七、面试要点

1. **堆优化是标配** — 面试中用邻接表+堆
2. **不能处理负权** — 必须知道这个限制
3. **路径还原** — 维护 prev 数组
4. **提前终止** — 找到目标即停止
