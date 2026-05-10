# 树链剖分 (Heavy Light Decomposition)

## 1. 概述

树链剖分（Heavy-Light Decomposition，HLD）是一种将树上的路径查询/修改问题转化为若干段连续区间的技巧。配合线段树或树状数组，可以在 O(log^2 n) 时间内完成树上路径操作。

## 2. 核心思想

将树的每条边分为**重边**和**轻边**：
- **重边**（Heavy Edge）：连接到子树最大的子节点的边
- **轻边**（Light Edge）：其他所有边

关键性质：从任意节点到根的路径上，经过的轻边数量不超过 O(log n)。

## 3. 重要概念

### 3.1 定义

| 概念 | 定义 |
|------|------|
| sz[u] | 以 u 为根的子树大小 |
| son[u] | u 的重子节点（子树最大的子节点） |
| dep[u] | u 的深度 |
| fa[u] | u 的父节点 |
| top[u] | u 所在重链的顶端节点 |
| dfn[u] | u 的DFS序编号 |
| rk[i] | DFS序为 i 的节点 |

### 3.2 重链剖分示意

```
树结构：         剖分后（粗线为重边）：

    1                1
   /|\              /|\
  2 3 4            2 3 4
 /|   |           /|   |
5 6   7          5 6   7
```

## 4. 实现步骤

### 4.1 第一次DFS：求 sz, son, dep, fa

```python
def dfs1(u, parent, depth):
    """第一次DFS：统计子树大小，找重子节点"""
    dep[u] = depth
    fa[u] = parent
    sz[u] = 1
    son[u] = -1  # 默认无重子节点
    max_sz = 0

    for v in graph[u]:
        if v == parent:
            continue
        dfs1(v, u, depth + 1)
        sz[u] += sz[v]

        if sz[v] > max_sz:
            max_sz = sz[v]
            son[u] = v  # 更新重子节点
```

### 4.2 第二次DFS：分配DFS序，确定重链

```python
def dfs2(u, top_node):
    """第二次DFS：分配dfn，确定top"""
    global timer
    timer += 1
    dfn[u] = timer
    rk[timer] = u
    top[u] = top_node

    if son[u] != -1:
        # 优先处理重子节点（延续重链）
        dfs2(son[u], top_node)

    for v in graph[u]:
        if v == fa[u] or v == son[u]:
            continue
        # 轻子节点开启新重链
        dfs2(v, v)
```

## 5. 树上路径查询

### 5.1 原理

查询路径 u-v 时，不断将深度较大的节点所在重链的顶端向上跳，直到 u 和 v 在同一重链上。

```python
def query_path(u, v):
    """查询路径 u-v 上的信息"""
    result = 0
    while top[u] != top[v]:
        # 将深度大的往上跳
        if dep[top[u]] < dep[top[v]]:
            u, v = v, u
        # 查询 [dfn[top[u]], dfn[u]] 这段区间
        result += seg_tree.query(dfn[top[u]], dfn[u])
        u = fa[top[u]]

    # 同一重链上
    if dep[u] > dep[v]:
        u, v = v, u
    result += seg_tree.query(dfn[u], dfn[v])
    return result
```

### 5.2 复杂度分析

每跳一次至少走一条轻边，轻边数 O(log n)，每次区间查询 O(log n)，总复杂度 O(log^2 n)。

## 6. 完整代码框架

```python
class HLD:
    """树链剖分"""

    def __init__(self, n, graph, values):
        self.n = n
        self.graph = graph
        self.values = values

        self.sz = [0] * (n + 1)
        self.son = [-1] * (n + 1)
        self.dep = [0] * (n + 1)
        self.fa = [0] * (n + 1)
        self.top = [0] * (n + 1)
        self.dfn = [0] * (n + 1)
        self.rk = [0] * (n + 1)

        self.timer = 0
        self.dfs1(1, 0, 1)
        self.dfs2(1, 1)

        # 建线段树
        arr = [values[rk[i]] for i in range(1, n + 1)]
        self.seg = SegmentTree(arr)

    def dfs1(self, u, parent, depth):
        self.dep[u] = depth
        self.fa[u] = parent
        self.sz[u] = 1
        max_sz = 0
        for v in self.graph[u]:
            if v == parent:
                continue
            self.dfs1(v, u, depth + 1)
            self.sz[u] += self.sz[v]
            if self.sz[v] > max_sz:
                max_sz = self.sz[v]
                self.son[u] = v

    def dfs2(self, u, top_node):
        self.timer += 1
        self.dfn[u] = self.timer
        self.rk[self.timer] = u
        self.top[u] = top_node
        if self.son[u] != -1:
            self.dfs2(self.son[u], top_node)
        for v in self.graph[u]:
            if v != self.fa[u] and v != self.son[u]:
                self.dfs2(v, v)

    def query_path(self, u, v):
        """查询路径u-v"""
        result = 0
        while self.top[u] != self.top[v]:
            if self.dep[self.top[u]] < self.dep[self.top[v]]:
                u, v = v, u
            result += self.seg.query(self.dfn[self.top[u]], self.dfn[u])
            u = self.fa[self.top[u]]
        if self.dep[u] > self.dep[v]:
            u, v = v, u
        result += self.seg.query(self.dfn[u], self.dfn[v])
        return result

    def update_node(self, u, val):
        """单点修改"""
        self.seg.update(self.dfn[u], val)
```

## 7. C++ 实现

```cpp
const int MAXN = 100005;
vector<int> graph[MAXN];
int sz[MAXN], son[MAXN], dep[MAXN], fa[MAXN];
int top[MAXN], dfn[MAXN], rk[MAXN];
int timer = 0;

void dfs1(int u, int parent, int depth) {
    dep[u] = depth;
    fa[u] = parent;
    sz[u] = 1;
    son[u] = -1;
    int maxSz = 0;
    for (int v : graph[u]) {
        if (v == parent) continue;
        dfs1(v, u, depth + 1);
        sz[u] += sz[v];
        if (sz[v] > maxSz) {
            maxSz = sz[v];
            son[u] = v;
        }
    }
}

void dfs2(int u, int topNode) {
    dfn[u] = ++timer;
    rk[timer] = u;
    top[u] = topNode;
    if (son[u] != -1)
        dfs2(son[u], topNode);
    for (int v : graph[u]) {
        if (v != fa[u] && v != son[u])
            dfs2(v, v);
    }
}

int queryPath(int u, int v) {
    int result = 0;
    while (top[u] != top[v]) {
        if (dep[top[u]] < dep[top[v]]) swap(u, v);
        result += query(1, 1, n, dfn[top[u]], dfn[u]);
        u = fa[top[u]];
    }
    if (dep[u] > dep[v]) swap(u, v);
    result += query(1, 1, n, dfn[u], dfn[v]);
    return result;
}
```

## 8. 应用场景

1. 树上路径求和/最大值
2. 树上路径修改
3. 子树查询（利用DFS序连续）
4. 最近公共祖先（LCA）

```python
def lca(self, u, v):
    """求LCA"""
    while self.top[u] != self.top[v]:
        if self.dep[self.top[u]] < self.dep[self.top[v]]:
            u, v = v, u
        u = self.fa[self.top[u]]
    return u if self.dep[u] < self.dep[v] else v
```

## 9. 总结

树链剖分将树上问题转化为序列问题：
- 两次DFS完成剖分，O(n)
- 路径查询/修改 O(log^2 n)
- 配合线段树或树状数组实现区间操作
- 代码量适中，实用性极强
