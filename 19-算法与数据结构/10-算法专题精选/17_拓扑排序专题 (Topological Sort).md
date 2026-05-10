# 拓扑排序专题 (Topological Sort)

## 一、概念定义与原理

### 1.1 拓扑排序

对**有向无环图 (DAG)** 的顶点进行线性排序，使得对于每条有向边 $(u, v)$，$u$ 在排序中出现在 $v$ 之前。

### 1.2 应用场景

- 课程安排（先修关系）
- 编译依赖
- 任务调度
- 判断有向图是否有环

### 1.3 存在性

拓扑排序存在当且仅当图是 DAG。如果存在环，则无法完成拓扑排序。

---

## 二、核心算法

### 2.1 Kahn 算法（BFS）

1. 计算所有节点的入度
2. 将入度为 0 的节点加入队列
3. 每次取出队首节点，将其邻居入度减 1
4. 若邻居入度变为 0，加入队列
5. 最终如果排序长度 < 节点数，说明存在环

### 2.2 DFS 算法

1. 对每个未访问节点 DFS
2. 递归结束后将节点加入栈（后序）
3. 最终栈的逆序即为拓扑排序
4. 若 DFS 遇到正在访问的节点，说明存在环

---

## 三、代码实现

### 3.1 Kahn 算法 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 返回拓扑排序，若存在环返回空
vector<int> topological_sort_kahn(vector<vector<int>>& graph, int n) {
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : graph[u]) indegree[v]++;

    queue<int> q;
    for (int i = 0; i < n; i++)
        if (indegree[i] == 0) q.push(i);

    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : graph[u]) {
            if (--indegree[v] == 0) q.push(v);
        }
    }
    return order.size() == n ? order : vector<int>();
}
```

### 3.2 DFS 算法 - C++

```cpp
bool dfs_topo(int u, vector<vector<int>>& graph,
              vector<int>& state, vector<int>& order) {
    state[u] = 1; // 正在访问
    for (int v : graph[u]) {
        if (state[v] == 1) return false; // 有环
        if (state[v] == 0 && !dfs_topo(v, graph, state, order)) return false;
    }
    state[u] = 2; // 访问完成
    order.push_back(u);
    return true;
}

vector<int> topological_sort_dfs(vector<vector<int>>& graph, int n) {
    vector<int> state(n, 0), order;
    for (int i = 0; i < n; i++) {
        if (state[i] == 0 && !dfs_topo(i, graph, state, order))
            return {}; // 有环
    }
    reverse(order.begin(), order.end());
    return order;
}
```

### 3.3 Python 实现

```python
from collections import deque

def topological_sort_kahn(graph, n):
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

def can_finish(num_courses, prerequisites):
    """LeetCode 207: 课程表"""
    graph = [[] for _ in range(num_courses)]
    indegree = [0] * num_courses
    for a, b in prerequisites:
        graph[b].append(a); indegree[a] += 1
    q = deque(i for i in range(num_courses) if indegree[i] == 0)
    count = 0
    while q:
        u = q.popleft(); count += 1
        for v in graph[u]:
            indegree[v] -= 1
            if indegree[v] == 0: q.append(v)
    return count == num_courses

# 测试
print(topological_sort_kahn([[], [0], [0], [1,2]], 4))  # [0,1,2,3] 或其他合法顺序
print(can_finish(2, [[1,0]]))  # True
```

### 3.4 字典序最小拓扑排序

```cpp
// 用优先队列代替普通队列，得到字典序最小的拓扑排序
vector<int> topo_lex_min(vector<vector<int>>& graph, int n) {
    vector<int> indegree(n, 0);
    for (int u = 0; u < n; u++)
        for (int v : graph[u]) indegree[v]++;

    priority_queue<int, vector<int>, greater<int>> pq;
    for (int i = 0; i < n; i++)
        if (indegree[i] == 0) pq.push(i);

    vector<int> order;
    while (!pq.empty()) {
        int u = pq.top(); pq.pop();
        order.push_back(u);
        for (int v : graph[u])
            if (--indegree[v] == 0) pq.push(v);
    }
    return order.size() == n ? order : vector<int>();
}
```

---

## 四、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| Kahn（BFS） | $O(V+E)$ | $O(V)$ |
| DFS | $O(V+E)$ | $O(V)$ |
| 字典序最小 | $O((V+E) \log V)$ | $O(V)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 207：** 课程表（判断是否有环）
2. **LeetCode 210：** 课程表II（输出排序）
3. **LeetCode 269：** 火星词典
4. **LeetCode 329：** 矩阵中的最长递增路径
5. **任务调度：** 确定任务执行顺序
