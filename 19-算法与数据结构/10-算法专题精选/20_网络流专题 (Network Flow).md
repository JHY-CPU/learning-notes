# 网络流专题 (Network Flow)

## 一、概念定义与原理

### 1.1 网络流基本概念

- **流网络：** 有向图，每条边有容量限制
- **源点 $s$、汇点 $t$：** 流入源点、流出汇点
- **可行流：** 满足容量约束和流量守恒的流
- **最大流：** 从源点到汇点的最大流量

### 1.2 增广路

从 $s$ 到 $t$ 的一条路径上，各边剩余容量的最小值称为这条路的**残余容量**。

### 1.3 最大流最小割定理

**定理：** 最大流 = 最小割

**最小割：** 将图分成 $S$ 和 $T$ 两个集合（$s \in S$, $t \in T$），使得从 $S$ 到 $T$ 的边权和最小。

---

## 二、核心算法

### 2.1 Ford-Fulkerson 方法

不断找增广路并沿路增广，直到不存在增广路。

### 2.2 Edmonds-Karp 算法

用 BFS 找最短增广路，时间复杂度 $O(VE^2)$。

### 2.3 Dinic 算法

在分层图上用 DFS 找阻塞流，时间复杂度 $O(V^2 E)$，实际效率很高。

---

## 三、代码实现

### 3.1 Dinic 算法 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct Edge { int to, rev; long long cap; };

class Dinic {
    vector<vector<Edge>> g;
    vector<int> level, iter;
    int n;
public:
    Dinic(int n) : n(n), g(n), level(n), iter(n) {}

    void add_edge(int from, int to, long long cap) {
        g[from].push_back({to, (int)g[to].size(), cap});
        g[to].push_back({from, (int)g[from].size()-1, 0}); // 反向边
    }

    bool bfs(int s, int t) {
        fill(level.begin(), level.end(), -1);
        queue<int> q;
        level[s] = 0; q.push(s);
        while (!q.empty()) {
            int v = q.front(); q.pop();
            for (auto& e : g[v]) {
                if (e.cap > 0 && level[e.to] < 0) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
        return level[t] >= 0;
    }

    long long dfs(int v, int t, long long f) {
        if (v == t) return f;
        for (int& i = iter[v]; i < g[v].size(); i++) {
            Edge& e = g[v][i];
            if (e.cap > 0 && level[v] < level[e.to]) {
                long long d = dfs(e.to, t, min(f, e.cap));
                if (d > 0) {
                    e.cap -= d;
                    g[e.to][e.rev].cap += d;
                    return d;
                }
            }
        }
        return 0;
    }

    long long max_flow(int s, int t) {
        long long flow = 0;
        while (bfs(s, t)) {
            fill(iter.begin(), iter.end(), 0);
            long long d;
            while ((d = dfs(s, t, LLONG_MAX)) > 0) flow += d;
        }
        return flow;
    }
};
```

### 3.2 Python 实现

```python
from collections import deque

class Dinic:
    def __init__(self, n):
        self.n = n
        self.graph = [[] for _ in range(n)]

    def add_edge(self, fr, to, cap):
        self.graph[fr].append([to, cap, len(self.graph[to])])
        self.graph[to].append([fr, 0, len(self.graph[fr]) - 1])

    def bfs(self, s, t):
        self.level = [-1] * self.n
        q = deque([s]); self.level[s] = 0
        while q:
            v = q.popleft()
            for to, cap, rev in self.graph[v]:
                if cap > 0 and self.level[to] < 0:
                    self.level[to] = self.level[v] + 1; q.append(to)
        return self.level[t] >= 0

    def dfs(self, v, t, f):
        if v == t: return f
        for i in range(self.it[v], len(self.graph[v])):
            self.it[v] = i
            to, cap, rev = self.graph[v][i]
            if cap > 0 and self.level[v] < self.level[to]:
                d = self.dfs(to, t, min(f, cap))
                if d > 0:
                    self.graph[v][i][1] -= d
                    self.graph[to][rev][1] += d
                    return d
        return 0

    def max_flow(self, s, t):
        flow = 0
        while self.bfs(s, t):
            self.it = [0] * self.n
            while True:
                f = self.dfs(s, t, float('inf'))
                if f == 0: break
                flow += f
        return flow

dinic = Dinic(4)
dinic.add_edge(0, 1, 3); dinic.add_edge(0, 2, 2)
dinic.add_edge(1, 3, 2); dinic.add_edge(2, 3, 3)
print(dinic.max_flow(0, 3))  # 4
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 说明 |
|------|-----------|------|
| Edmonds-Karp | $O(VE^2)$ | BFS 增广 |
| Dinic | $O(V^2 E)$ | 分层+阻塞流 |
| ISAP | $O(V^2 E)$ | 实际效率高 |

---

## 五、竞赛与面试应用场景

1. **最大流：** LeetCode 及 Codeforces 网络流题
2. **最小割：** 图像分割、项目选择
3. **二分图匹配：** 转化为最大流
4. **费用流：** 最小费用最大流
5. **多重匹配：** 带容量的匹配问题
