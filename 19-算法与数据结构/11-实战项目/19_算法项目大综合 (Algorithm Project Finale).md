# 算法项目综合 (Algorithm Project Finale)

## 项目需求与功能分析

作为算法实战系列的总结项目，本项目综合运用排序、搜索、图论、动态规划等多种算法，实现一个简易的智能路线规划系统。系统模拟一个城市交通网络，支持最短路径查询、拥堵分析、公交线路优化等功能。

### 核心功能

- 城市交通图建模（节点 = 站点，边 = 道路）
- 多种最短路径算法（Dijkstra、A*、Floyd）
- 道路拥堵模拟与动态权重调整
- 公交线路贪心优化
- 出行方案排序（综合时间、换乘、费用）
- 统计分析（路网连通性、平均通行时间）

### 综合运用的算法

| 算法 | 应用场景 |
|------|----------|
| Dijkstra / A* | 单源最短路径 |
| Floyd-Warshall | 全源最短路径 |
| 贪心算法 | 公交线路规划 |
| 排序算法 | 方案排序 |
| 并查集 | 连通性检测 |
| 动态规划 | 最优换乘方案 |
| BFS / DFS | 路网遍历 |

## 完整代码实现

```python
import heapq
import math
import random
from collections import defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field


@dataclass
class Edge:
    """道路"""
    to: str
    distance: float  # 公里
    time: float      # 分钟
    cost: float      # 费用
    congestion: float = 0.0  # 拥堵程度 0~1

    @property
    def effective_time(self):
        return self.time * (1 + self.congestion)


@dataclass
class Route:
    """出行方案"""
    path: List[str]
    total_distance: float
    total_time: float
    total_cost: float
    transfers: int  # 换乘次数

    def score(self, w_time=0.5, w_cost=0.3, w_transfer=0.2) -> float:
        """综合评分"""
        return (w_time * self.total_time
                + w_cost * self.total_cost * 10
                + w_transfer * self.transfers * 5)


class CityGraph:
    """城市交通图"""

    def __init__(self):
        self.graph: Dict[str, List[Edge]] = defaultdict(list)
        self.stations: Set[str] = set()
        self.coordinates: Dict[str, Tuple[float, float]] = {}

    def add_station(self, name: str, x: float = 0, y: float = 0):
        self.stations.add(name)
        self.coordinates[name] = (x, y)

    def add_road(self, u: str, v: str, distance: float,
                  time: float, cost: float, bidirectional: bool = True):
        self.graph[u].append(Edge(v, distance, time, cost))
        self.stations.add(u); self.stations.add(v)
        if bidirectional:
            self.graph[v].append(Edge(u, distance, time, cost))

    def set_congestion(self, u: str, v: str, level: float):
        """设置道路拥堵程度"""
        for edge in self.graph[u]:
            if edge.to == v:
                edge.congestion = level
        for edge in self.graph[v]:
            if edge.to == u:
                edge.congestion = level

    def simulate_congestion(self):
        """随机模拟拥堵"""
        for u in self.graph:
            for edge in self.graph[u]:
                edge.congestion = random.uniform(0, 0.8)

    # ===== 最短路径算法 =====

    def dijkstra(self, start: str, end: str,
                  weight: str = 'time') -> Tuple[float, List[str]]:
        """Dijkstra 最短路径"""
        dist = {s: float('inf') for s in self.stations}
        dist[start] = 0
        parent = {start: None}
        heap = [(0, start)]

        while heap:
            d, u = heapq.heappop(heap)
            if u == end:
                break
            if d > dist[u]:
                continue
            for edge in self.graph[u]:
                w = getattr(edge, 'effective_time' if weight == 'time' else weight)
                nd = d + w
                if nd < dist[edge.to]:
                    dist[edge.to] = nd
                    parent[edge.to] = u
                    heapq.heappush(heap, (nd, edge.to))

        path = self._reconstruct(parent, start, end)
        return dist[end], path

    def astar(self, start: str, end: str) -> Tuple[float, List[str]]:
        """A* 搜索"""
        def heuristic(a, b):
            if a in self.coordinates and b in self.coordinates:
                ax, ay = self.coordinates[a]
                bx, by = self.coordinates[b]
                return math.sqrt((ax-bx)**2 + (ay-by)**2) / 50 * 60  # 估算时间
            return 0

        g = {s: float('inf') for s in self.stations}
        g[start] = 0
        parent = {start: None}
        heap = [(heuristic(start, end), start)]
        visited = set()

        while heap:
            f, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            if u == end:
                break
            for edge in self.graph[u]:
                ng = g[u] + edge.effective_time
                if ng < g[edge.to]:
                    g[edge.to] = ng
                    parent[edge.to] = u
                    f_score = ng + heuristic(edge.to, end)
                    heapq.heappush(heap, (f_score, edge.to))

        path = self._reconstruct(parent, start, end)
        return g[end], path

    def floyd_warshall(self) -> Dict[Tuple[str, str], float]:
        """Floyd 全源最短路径"""
        stations = sorted(self.stations)
        n = len(stations)
        idx = {s: i for i, s in enumerate(stations)}
        INF = float('inf')
        dist = [[INF] * n for _ in range(n)]

        for i in range(n): dist[i][i] = 0
        for u in self.graph:
            for edge in self.graph[u]:
                i, j = idx[u], idx[edge.to]
                dist[i][j] = min(dist[i][j], edge.effective_time)

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        result = {}
        for i, si in enumerate(stations):
            for j, sj in enumerate(stations):
                result[(si, sj)] = dist[i][j]
        return result

    def _reconstruct(self, parent, start, end):
        if end not in parent and end != start:
            return []
        path = []
        node = end
        while node is not None:
            path.append(node)
            node = parent.get(node)
        path.reverse()
        return path if path[0] == start else []

    # ===== 网络分析 =====

    def connected_components(self) -> List[List[str]]:
        """连通分量"""
        visited = set()
        components = []
        for station in self.stations:
            if station not in visited:
                comp = []
                queue = deque([station])
                while queue:
                    node = queue.popleft()
                    if node in visited:
                        continue
                    visited.add(node)
                    comp.append(node)
                    for edge in self.graph[node]:
                        if edge.to not in visited:
                            queue.append(edge.to)
                components.append(comp)
        return components

    def find_routes(self, start: str, end: str,
                     max_routes: int = 3) -> List[Route]:
        """找多条出行方案"""
        routes = []

        # 方案1: 最快路线
        time_val, time_path = self.dijkstra(start, end, 'time')
        if time_path:
            d, t, c = self._path_stats(time_path)
            routes.append(Route(time_path, d, t, c, self._count_transfers(time_path)))

        # 方案2: 最短距离
        dist_val, dist_path = self.dijkstra(start, end, 'distance')
        if dist_path and dist_path != time_path:
            d, t, c = self._path_stats(dist_path)
            routes.append(Route(dist_path, d, t, c, self._count_transfers(dist_path)))

        # 方案3: 最便宜
        cost_val, cost_path = self.dijkstra(start, end, 'cost')
        if cost_path and cost_path not in [r.path for r in routes]:
            d, t, c = self._path_stats(cost_path)
            routes.append(Route(cost_path, d, t, c, self._count_transfers(cost_path)))

        # 按综合评分排序
        routes.sort(key=lambda r: r.score())
        return routes[:max_routes]

    def _path_stats(self, path):
        d = t = c = 0.0
        for i in range(len(path) - 1):
            for edge in self.graph[path[i]]:
                if edge.to == path[i+1]:
                    d += edge.distance
                    t += edge.effective_time
                    c += edge.cost
                    break
        return d, t, c

    def _count_transfers(self, path):
        return max(0, len(path) - 2)

    def display_network(self):
        print(f"\n{'='*50}")
        print(f"城市交通网络 (站点: {len(self.stations)})")
        print(f"{'='*50}")
        for station in sorted(self.stations):
            edges = self.graph[station]
            print(f"  {station}: {len(edges)} 条连接")
            for e in edges:
                cong = f" [拥堵:{e.congestion:.0%}]" if e.congestion > 0 else ""
                print(f"    -> {e.to}: {e.distance}km, {e.effective_time:.1f}min, ¥{e.cost}{cong}")


def demo():
    city = CityGraph()
    # 添加站点
    stations = {
        '火车站': (0, 0), '市中心': (3, 2), '科技园': (6, 1),
        '大学城': (2, 5), '体育馆': (5, 4), '机场': (8, 3),
        '医院': (1, 3), '公园': (4, 6),
    }
    for name, (x, y) in stations.items():
        city.add_station(name, x, y)

    # 添加道路
    roads = [
        ('火车站', '市中心', 3, 10, 2),
        ('市中心', '科技园', 4, 12, 3),
        ('火车站', '大学城', 5, 15, 2),
        ('大学城', '体育馆', 3, 8, 2),
        ('体育馆', '机场', 4, 10, 5),
        ('科技园', '机场', 3, 8, 5),
        ('市中心', '医院', 2, 6, 1),
        ('医院', '大学城', 3, 10, 1),
        ('大学城', '公园', 2, 5, 1),
        ('公园', '体育馆', 2, 6, 1),
    ]
    for u, v, d, t, c in roads:
        city.add_road(u, v, d, t, c)

    city.display_network()

    # 最短路径查询
    print("\n--- 最短路径查询: 火车站 -> 机场 ---")
    for algo_name, algo_fn in [('Dijkstra', city.dijkstra), ('A*', city.astar)]:
        val, path = algo_fn('火车站', '机场')
        print(f"  {algo_name}: 时间={val:.1f}min, 路径={' -> '.join(path)}")

    # 多方案推荐
    print("\n--- 出行方案推荐: 火车站 -> 机场 ---")
    routes = city.find_routes('火车站', '机场')
    for i, r in enumerate(routes, 1):
        print(f"  方案{i}: {'->'.join(r.path)}")
        print(f"    距离={r.total_distance:.1f}km, 时间={r.total_time:.1f}min, "
              f"费用=¥{r.total_cost:.0f}, 评分={r.score():.1f}")

    # 连通性分析
    comps = city.connected_components()
    print(f"\n连通分量: {len(comps)} 个")


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestCityGraph(unittest.TestCase):
    def setUp(self):
        self.city = CityGraph()
        for s in ['A','B','C','D']: self.city.add_station(s)
        self.city.add_road('A','B',2,5,1)
        self.city.add_road('B','C',3,8,2)
        self.city.add_road('A','C',10,20,5)
        self.city.add_road('C','D',1,3,1)

    def test_dijkstra(self):
        val, path = self.city.dijkstra('A', 'C')
        self.assertEqual(path, ['A','B','C'])
        self.assertAlmostEqual(val, 13)

    def test_astar(self):
        val, path = self.city.astar('A', 'C')
        self.assertIsNotNone(path)
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'C')

    def test_floyd(self):
        dist = self.city.floyd_warshall()
        self.assertAlmostEqual(dist[('A','C')], 13)
        self.assertAlmostEqual(dist[('A','D')], 16)

    def test_connected(self):
        comps = self.city.connected_components()
        self.assertEqual(len(comps), 1)

    def test_find_routes(self):
        routes = self.city.find_routes('A', 'D')
        self.assertGreater(len(routes), 0)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **实时路况**：接入真实的交通数据 API
2. **公交换乘**：模拟公交线路，优化换乘方案
3. **步行 / 骑行**：多出行方式混合规划
4. **时间窗口**：考虑首末班车时间约束
5. **图数据库**：使用 Neo4j 存储路网数据
6. **地图可视化**：在地图上展示路线和拥堵
7. **机器学习**：基于历史数据预测拥堵趋势
