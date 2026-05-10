# 图编辑器 (Graph Editor)

## 项目需求与功能分析

图是算法和数据结构中最重要的抽象数据类型之一。本项目实现一个交互式图编辑器，支持图的创建、编辑、可视化以及常见图算法的执行。

### 核心功能

- 交互式添加 / 删除节点和边
- 支持有向图和无向图
- 支持带权边
- 图算法执行（BFS、DFS、最短路径、最小生成树、拓扑排序）
- 邻接矩阵和邻接表可视化
- 图的统计信息（度、连通分量等）

## 核心算法原理

### 图的表示

**邻接矩阵**：G[i][j] 表示节点 i 到节点 j 的边权。空间 O(V^2)，适合稠密图。

**邻接表**：每个节点维护一个邻居列表。空间 O(V+E)，适合稀疏图。

### 最小生成树 - Kruskal

1. 将所有边按权重排序
2. 依次选择权重最小的边，若加入后不形成环则加入
3. 使用并查集检测环

### 拓扑排序 - Kahn

1. 计算所有节点的入度
2. 将入度为 0 的节点加入队列
3. 取出队首节点，将其邻居入度减 1，若为 0 则入队
4. 重复直到队列为空

## 完整代码实现

```python
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Set, Optional
import heapq


class Graph:
    """图数据结构 - 邻接表实现"""

    def __init__(self, directed=False, weighted=False):
        self.directed = directed
        self.weighted = weighted
        self.adj: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
        self.nodes: Set[int] = set()
        self.edges: List[Tuple[int, int, float]] = []

    def add_node(self, node: int):
        self.nodes.add(node)

    def add_edge(self, u: int, v: int, weight: float = 1.0):
        self.nodes.add(u); self.nodes.add(v)
        self.adj[u].append((v, weight))
        if not self.directed:
            self.adj[v].append((u, weight))
        self.edges.append((u, v, weight))

    def remove_edge(self, u: int, v: int):
        self.adj[u] = [(n, w) for n, w in self.adj[u] if n != v]
        if not self.directed:
            self.adj[v] = [(n, w) for n, w in self.adj[v] if n != u]

    def remove_node(self, node: int):
        self.nodes.discard(node)
        del self.adj[node]
        for n in self.adj:
            self.adj[n] = [(v, w) for v, w in self.adj[n] if v != node]

    def degree(self, node: int) -> int:
        return len(self.adj[node])

    def in_degree(self, node: int) -> int:
        if not self.directed:
            return self.degree(node)
        return sum(1 for n in self.adj for v, _ in self.adj[n] if v == node)

    def neighbors(self, node: int) -> List[Tuple[int, float]]:
        return self.adj.get(node, [])

    def bfs(self, start: int) -> List[int]:
        visited, order = {start}, []
        queue = deque([start])
        while queue:
            node = queue.popleft(); order.append(node)
            for nb, _ in self.adj[node]:
                if nb not in visited:
                    visited.add(nb); queue.append(nb)
        return order

    def dfs(self, start: int) -> List[int]:
        visited, order = set(), []
        def _dfs(node):
            visited.add(node); order.append(node)
            for nb, _ in self.adj[node]:
                if nb not in visited: _dfs(nb)
        _dfs(start)
        return order

    def dijkstra(self, start: int) -> Tuple[Dict[int, float], Dict[int, Optional[int]]]:
        dist = {n: float('inf') for n in self.nodes}
        dist[start] = 0
        parent = {start: None}
        heap = [(0, start)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]: continue
            for v, w in self.adj[u]:
                nd = d + w
                if nd < dist[v]:
                    dist[v] = nd; parent[v] = u
                    heapq.heappush(heap, (nd, v))
        return dist, parent

    def shortest_path(self, start: int, end: int) -> Tuple[float, List[int]]:
        dist, parent = self.dijkstra(start)
        if dist[end] == float('inf'):
            return float('inf'), []
        path = []
        node = end
        while node is not None:
            path.append(node); node = parent.get(node)
        path.reverse()
        return dist[end], path

    def kruskal_mst(self) -> List[Tuple[int, int, float]]:
        """Kruskal 最小生成树"""
        parent = {n: n for n in self.nodes}
        rank = {n: 0 for n in self.nodes}

        def find(x):
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb: return False
            if rank[ra] < rank[rb]: ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]: rank[ra] += 1
            return True

        sorted_edges = sorted(self.edges, key=lambda e: e[2])
        mst = []
        for u, v, w in sorted_edges:
            if union(u, v):
                mst.append((u, v, w))
                if len(mst) == len(self.nodes) - 1:
                    break
        return mst

    def topological_sort(self) -> Optional[List[int]]:
        """Kahn 拓扑排序"""
        if not self.directed:
            return None
        in_deg = defaultdict(int)
        for n in self.nodes: in_deg[n]  # 确保每个节点都有记录
        for u in self.adj:
            for v, _ in self.adj[u]:
                in_deg[v] += 1
        queue = deque([n for n in self.nodes if in_deg[n] == 0])
        order = []
        while queue:
            node = queue.popleft(); order.append(node)
            for nb, _ in self.adj[node]:
                in_deg[nb] -= 1
                if in_deg[nb] == 0: queue.append(nb)
        return order if len(order) == len(self.nodes) else None

    def connected_components(self) -> List[List[int]]:
        """连通分量"""
        visited = set()
        components = []
        for node in self.nodes:
            if node not in visited:
                comp = self.bfs(node)
                visited.update(comp)
                components.append(comp)
        return components

    def adjacency_matrix(self) -> List[List[float]]:
        nodes = sorted(self.nodes)
        idx = {n: i for i, n in enumerate(nodes)}
        size = len(nodes)
        matrix = [[0.0] * size for _ in range(size)]
        for u in self.adj:
            for v, w in self.adj[u]:
                matrix[idx[u]][idx[v]] = w
        return matrix

    def display(self):
        print(f"\n图类型: {'有向' if self.directed else '无向'}{'加权' if self.weighted else ''}")
        print(f"节点: {sorted(self.nodes)}")
        print(f"边数: {len(self.edges)}")
        print("邻接表:")
        for node in sorted(self.nodes):
            nbs = [(v, w) for v, w in self.adj[node]]
            print(f"  {node} -> {nbs}")


def demo():
    g = Graph(directed=True, weighted=True)
    edges = [(0,1,4),(0,2,1),(1,3,1),(2,1,2),(2,3,5),(3,4,3)]
    for u,v,w in edges: g.add_edge(u,v,w)
    g.display()

    print("\nBFS 从 0:", g.bfs(0))
    print("DFS 从 0:", g.dfs(0))

    dist, _ = g.dijkstra(0)
    print("\nDijkstra 最短距离:", dist)

    d, path = g.shortest_path(0, 4)
    print(f"0->4 最短路径: 距离={d}, 路径={path}")

    topo = g.topological_sort()
    print(f"拓扑排序: {topo}")


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestGraph(unittest.TestCase):
    def test_bfs_dfs(self):
        g = Graph()
        for u,v in [(0,1),(0,2),(1,3),(2,3)]: g.add_edge(u,v)
        self.assertEqual(g.bfs(0), [0,1,2,3])

    def test_dijkstra(self):
        g = Graph(weighted=True)
        g.add_edge(0,1,1); g.add_edge(0,2,4); g.add_edge(1,2,2)
        dist, _ = g.dijkstra(0)
        self.assertEqual(dist[0], 0)
        self.assertEqual(dist[2], 3)  # 0->1->2

    def test_mst(self):
        g = Graph(weighted=True)
        g.add_edge(0,1,1); g.add_edge(1,2,2); g.add_edge(0,2,3)
        mst = g.kruskal_mst()
        self.assertEqual(len(mst), 2)
        self.assertEqual(sum(w for _,_,w in mst), 3)

    def test_topo_sort(self):
        g = Graph(directed=True)
        g.add_edge(0,1); g.add_edge(0,2); g.add_edge(1,3); g.add_edge(2,3)
        topo = g.topological_sort()
        self.assertIsNotNone(topo)
        self.assertEqual(set(topo), {0,1,2,3})

    def test_cycle_no_topo(self):
        g = Graph(directed=True)
        g.add_edge(0,1); g.add_edge(1,0)
        self.assertIsNone(g.topological_sort())

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **图形化界面**：使用 Pygame 或 Tkinter 实现节点拖拽
2. **更多算法**：Bellman-Ford、Floyd-Warshall、网络流
3. **图的导入导出**：支持 DOT、GraphML 等格式
4. **动画演示**：逐步展示算法执行过程
5. **强连通分量**：Tarjan / Kosaraju 算法
6. **图着色**：贪心图着色算法
7. **JSON 序列化**：将图结构持久化为 JSON
