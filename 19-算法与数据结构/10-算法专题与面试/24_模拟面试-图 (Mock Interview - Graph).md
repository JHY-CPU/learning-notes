# 模拟面试-图 (Mock Interview - Graph)

## 一、面试流程模拟

**时间：** 45分钟
**重点：** 图的表示、BFS/DFS、拓扑排序、并查集

---

## 二、题目1：岛屿数量 (LeetCode 200, Medium, 15分钟)

### 面试过程

**候选人：**
"遍历整个网格，每发现一个 '1' 就启动 DFS/BFS 把整个岛标记为已访问，计数加一。"

### 代码

```python
def num_islands(grid):
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    count = 0

    def dfs(i, j):
        if i < 0 or i >= m or j < 0 or j >= n or grid[i][j] != '1':
            return
        grid[i][j] = '#'  # 标记已访问
        dfs(i+1, j); dfs(i-1, j)
        dfs(i, j+1); dfs(i, j-1)

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                count += 1
                dfs(i, j)
    return count
```

**面试官追问：** "不想修改原数组呢？"
"用一个额外的 visited 二维数组或集合记录访问过的坐标。"

**复杂度：** 时间 $O(mn)$，空间 $O(mn)$（最坏情况递归栈）。

---

## 三、题目2：课程表II (LeetCode 210, Medium, 15分钟)

### 题目描述

给定课程数和先修关系，返回学完所有课程的一种顺序。不可能则返回空。

### 面试过程

**候选人：**
"拓扑排序。BFS (Kahn) 算法：维护入度数组，从入度为0的节点开始BFS。"

### 代码

```python
def find_order(num_courses, prerequisites):
    graph = [[] for _ in range(num_courses)]
    in_degree = [0] * num_courses

    for a, b in prerequisites:
        graph[b].append(a)
        in_degree[a] += 1

    queue = [i for i in range(num_courses) if in_degree[i] == 0]
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == num_courses else []
```

---

## 四、题目3：冗余连接 (LeetCode 684, Medium, 15分钟)

### 面试过程

**候选人：**
"无向图中找导致环的边。用并查集，每加一条边检查两个端点是否已经在同一个集合中。如果是，这条边就是冗余的。"

### 代码

```python
def find_redundant_connection(edges):
    parent = list(range(len(edges) + 1))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[py] = px
        return True

    for u, v in edges:
        if not union(u, v):
            return [u, v]
```

**面试官追问：** "路径压缩和按秩合并的区别？"
"路径压缩在 find 时将节点直接连到根，按秩合并在 union 时将矮树挂到高树下。两者都是为优化到近 O(1)。"

---

## 五、评分要点

1. **图的建模能力** — 能否将问题转化为图论模型
2. **BFS/DFS熟练度** — 能否快速写出遍历代码
3. **并查集** — 是否掌握路径压缩
4. **拓扑排序** — 理解入度和 Kahn 算法
