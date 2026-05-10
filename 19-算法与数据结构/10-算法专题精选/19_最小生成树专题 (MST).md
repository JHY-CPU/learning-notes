# 最小生成树专题 (Minimum Spanning Tree, MST)

## 一、概念定义与原理

### 1.1 生成树

对于有 $n$ 个顶点的连通无向图，**生成树**是包含所有 $n$ 个顶点的 $n-1$ 条边的连通子图，且无环。

### 1.2 最小生成树

边权之和最小的生成树。

### 1.3 MST 性质

1. **唯一性：** MST 不一定唯一，但若所有边权不同则唯一
2. **切割性质：** 对于任意切割，横跨切割的最小权边一定在某个 MST 中
3. **环路性质：** 对于任意环，环上最大权边不在任何 MST 中

---

## 二、核心算法

### 2.1 Kruskal 算法

1. 将所有边按权值排序
2. 依次取最小边，如果两端不在同一集合（并查集），则加入 MST
3. 重复直到选了 $n-1$ 条边

时间复杂度：$O(E \log E)$（瓶颈在排序）

### 2.2 Prim 算法

类似 Dijkstra，从任意点出发，每次将距离 MST 最近的点加入。

- 朴素：$O(V^2)$，适合稠密图
- 堆优化：$O(E \log V)$，适合稀疏图

---

## 三、代码实现

### 3.1 Kruskal - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct DSU {
    vector<int> fa;
    DSU(int n) : fa(n) { iota(fa.begin(), fa.end(), 0); }
    int find(int x) { return fa[x] == x ? x : fa[x] = find(fa[x]); }
    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        fa[y] = x; return true;
    }
};

long long kruskal(int n, vector<tuple<long long,int,int>>& edges) {
    sort(edges.begin(), edges.end());
    DSU dsu(n);
    long long cost = 0;
    int cnt = 0;
    for (auto [w, u, v] : edges) {
        if (dsu.unite(u, v)) {
            cost += w;
            if (++cnt == n - 1) break;
        }
    }
    return cnt == n - 1 ? cost : -1; // -1 表示不连通
}
```

### 3.2 Prim（堆优化）- C++

```cpp
long long prim(vector<vector<pair<int,long long>>>& g) {
    int n = g.size();
    vector<bool> in_mst(n, false);
    priority_queue<pair<long long,int>, vector<pair<long long,int>>, greater<>> pq;
    pq.push({0, 0});
    long long cost = 0;
    int cnt = 0;
    while (!pq.empty() && cnt < n) {
        auto [w, u] = pq.top(); pq.pop();
        if (in_mst[u]) continue;
        in_mst[u] = true;
        cost += w;
        cnt++;
        for (auto [v, wt] : g[u]) {
            if (!in_mst[v]) pq.push({wt, v});
        }
    }
    return cnt == n ? cost : -1;
}
```

### 3.3 Python 实现

```python
class DSU:
    def __init__(self, n):
        self.fa = list(range(n))
    def find(self, x):
        if self.fa[x] != x: self.fa[x] = self.find(self.fa[x])
        return self.fa[x]
    def unite(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        self.fa[ry] = rx; return True

def kruskal(n, edges):
    edges.sort()
    dsu = DSU(n); cost = 0; cnt = 0
    for w, u, v in edges:
        if dsu.unite(u, v):
            cost += w; cnt += 1
            if cnt == n - 1: break
    return cost if cnt == n - 1 else -1

import heapq
def prim(graph, n):
    in_mst = [False] * n; pq = [(0, 0)]; cost = 0; cnt = 0
    while pq and cnt < n:
        w, u = heapq.heappop(pq)
        if in_mst[u]: continue
        in_mst[u] = True; cost += w; cnt += 1
        for v, wt in graph[u]:
            if not in_mst[v]: heapq.heappush(pq, (wt, v))
    return cost if cnt == n else -1

# 测试
edges = [(4,0,1), (1,0,2), (2,1,2), (3,1,3), (5,2,3)]
print(kruskal(4, edges))  # 6 (边 0-2, 1-2, 1-3)
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 | 适用场景 |
|------|-----------|-----------|---------|
| Kruskal | $O(E \log E)$ | $O(V)$ | 稀疏图 |
| Prim（朴素） | $O(V^2)$ | $O(V)$ | 稠密图 |
| Prim（堆优化） | $O(E \log V)$ | $O(V)$ | 稀疏图 |

---

## 五、竞赛与面试应用场景

1. **LeetCode 1135：** 最低成本联通所有城市
2. **LeetCode 1584：** 连接所有点的最小费用
3. **网络设计：** 用最小成本连接所有节点
4. **次小生成树：** 在 MST 基础上替换一条边
5. **瓶颈生成树：** 最大边权最小的生成树
