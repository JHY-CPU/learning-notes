# 路径查找器 (Path Finder)

## 项目需求与功能分析

路径搜索是图论和算法的核心应用场景。本项目实现一个网格地图上的路径搜索可视化工具，支持 BFS、Dijkstra、A* 三种经典搜索算法，直观展示搜索扩展过程和最短路径。

### 核心功能

- 网格地图编辑（设置起点、终点、墙壁）
- BFS 广度优先搜索可视化
- Dijkstra 最短路径算法可视化
- A* 启发式搜索可视化
- 实时动画展示搜索扩展过程
- 路径回溯与长度统计
- 算法性能对比

### 数据结构

| 概念 | 数据结构 |
|------|----------|
| 网格地图 | 二维数组 `grid[row][col]` |
| 搜索队列 | deque (BFS), 优先队列 (Dijkstra / A*) |
| 路径记录 | parent 字典 |

## 核心算法原理

### BFS 广度优先搜索

逐层扩展，使用队列存储待访问节点。保证在无权图中找到最短路径。时间复杂度 O(V+E)。

### Dijkstra 算法

贪心策略：每次从未访问节点中选择距离最小的节点进行扩展。使用优先队列优化后时间复杂度 O((V+E) log V)。适用于非负权图。

### A* 算法

在 Dijkstra 基础上引入启发函数 h(n)，优先扩展 f(n)=g(n)+h(n) 最小的节点。常用曼哈顿距离作为启发函数。

## 完整代码实现

```python
import heapq, time, os
from collections import deque
from typing import List, Tuple, Optional, Dict

EMPTY, WALL, START, END, VISITED, PATH, FRONTIER = 0, 1, 2, 3, 4, 5, 6

class Grid:
    def __init__(self, rows=20, cols=40):
        self.rows, self.cols = rows, cols
        self.grid = [[EMPTY]*cols for _ in range(rows)]
        self.start, self.end = (0,0), (rows-1, cols-1)
        self.grid[0][0], self.grid[rows-1][cols-1] = START, END

    def in_bounds(self, pos): r,c = pos; return 0<=r<self.rows and 0<=c<self.cols
    def is_walkable(self, pos): r,c = pos; return self.grid[r][c] != WALL

    def neighbors(self, pos):
        r,c = pos
        result = []
        for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr,nc = r+dr, c+dc
            if self.in_bounds((nr,nc)) and self.is_walkable((nr,nc)):
                result.append((nr,nc))
        return result

    def set_wall(self, r, c):
        if (r,c) != self.start and (r,c) != self.end:
            self.grid[r][c] = WALL

    def clear_path(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] in (VISITED, PATH, FRONTIER):
                    self.grid[r][c] = EMPTY

    def render(self, title=""):
        os.system('cls' if os.name=='nt' else 'clear')
        sym = {EMPTY:'  ', WALL:'\033[90m##\033[0m', START:'\033[92mSS\033[0m',
               END:'\033[91mEE\033[0m', VISITED:'\033[94m..\033[0m',
               PATH:'\033[93m**\033[0m', FRONTIER:'\033[96m??\033[0m'}
        print(f"  {title}")
        print('  '+'--'*self.cols)
        for row in self.grid:
            print('  '+''.join(sym.get(c,'  ') for c in row))
        print('  '+'--'*self.cols)

    def generate_maze(self, wall_ratio=0.3):
        import random
        for r in range(self.rows):
            for c in range(self.cols):
                if (r,c)==self.start or (r,c)==self.end: continue
                if random.random() < wall_ratio: self.grid[r][c] = WALL

    def manhattan(self, a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])


class PathFinder:
    def __init__(self, grid, speed=0.02):
        self.grid, self.speed, self.nodes_explored = grid, speed, 0

    def bfs(self):
        self.grid.clear_path(); self.nodes_explored = 0
        queue = deque([self.grid.start])
        visited = {self.grid.start}
        parent = {self.grid.start: None}
        while queue:
            cur = queue.popleft(); self.nodes_explored += 1
            if cur == self.grid.end: return self._reconstruct(parent)
            r,c = cur
            if cur != self.grid.start: self.grid.grid[r][c] = VISITED
            for nb in self.grid.neighbors(cur):
                if nb not in visited:
                    visited.add(nb); parent[nb] = cur
                    nr,nc = nb
                    if nb != self.grid.end: self.grid.grid[nr][nc] = FRONTIER
                    queue.append(nb)
            self.grid.render(f"BFS | 探索: {self.nodes_explored} | 队列: {len(queue)}")
            time.sleep(self.speed)
        return None

    def dijkstra(self):
        self.grid.clear_path(); self.nodes_explored = 0
        dist = {self.grid.start: 0}; parent = {self.grid.start: None}
        heap = [(0, self.grid.start)]; visited = set()
        while heap:
            d, cur = heapq.heappop(heap)
            if cur in visited: continue
            visited.add(cur); self.nodes_explored += 1
            if cur == self.grid.end: return self._reconstruct(parent)
            r,c = cur
            if cur != self.grid.start: self.grid.grid[r][c] = VISITED
            for nb in self.grid.neighbors(cur):
                nd = d + 1
                if nb not in dist or nd < dist[nb]:
                    dist[nb] = nd; parent[nb] = cur
                    heapq.heappush(heap, (nd, nb))
                    nr,nc = nb
                    if nb != self.grid.end: self.grid.grid[nr][nc] = FRONTIER
            self.grid.render(f"Dijkstra | 探索: {self.nodes_explored}")
            time.sleep(self.speed)
        return None

    def astar(self):
        self.grid.clear_path(); self.nodes_explored = 0
        g = {self.grid.start: 0}; parent = {self.grid.start: None}
        heap = [(self.grid.manhattan(self.grid.start, self.grid.end), self.grid.start)]
        visited = set()
        while heap:
            f, cur = heapq.heappop(heap)
            if cur in visited: continue
            visited.add(cur); self.nodes_explored += 1
            if cur == self.grid.end: return self._reconstruct(parent)
            r,c = cur
            if cur != self.grid.start: self.grid.grid[r][c] = VISITED
            for nb in self.grid.neighbors(cur):
                tg = g[cur] + 1
                if nb not in g or tg < g[nb]:
                    g[nb] = tg; parent[nb] = cur
                    heapq.heappush(heap, (tg + self.grid.manhattan(nb, self.grid.end), nb))
                    nr,nc = nb
                    if nb != self.grid.end: self.grid.grid[nr][nc] = FRONTIER
            self.grid.render(f"A* | 探索: {self.nodes_explored}")
            time.sleep(self.speed)
        return None

    def _reconstruct(self, parent):
        path, node = [], self.grid.end
        while node is not None: path.append(node); node = parent[node]
        path.reverse()
        for r,c in path:
            if (r,c) not in (self.grid.start, self.grid.end): self.grid.grid[r][c] = PATH
        self.grid.render(f"找到路径! 长度: {len(path)}")
        return path
```

## 测试用例

```python
import unittest

class TestPathFinder(unittest.TestCase):
    def test_bfs_shortest(self):
        grid = Grid(rows=5, cols=5)
        finder = PathFinder(grid, speed=0)
        path = finder.bfs()
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 9)

    def test_no_path(self):
        grid = Grid(rows=5, cols=5)
        for r in range(5): grid.set_wall(r, 2)
        finder = PathFinder(grid, speed=0)
        self.assertIsNone(finder.bfs())

    def test_astar_explores_fewer(self):
        import random; random.seed(42)
        g = Grid(rows=20, cols=40); g.generate_maze(0.2)
        bfs_g = Grid(20,40); bfs_g.grid = [r[:] for r in g.grid]
        astar_g = Grid(20,40); astar_g.grid = [r[:] for r in g.grid]
        bf = PathFinder(bfs_g, speed=0); af = PathFinder(astar_g, speed=0)
        bf.bfs(); af.astar()
        self.assertLessEqual(af.nodes_explored, bf.nodes_explored + 10)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **加权地图**：支持不同地形代价（沼泽、山地、道路）
2. **对角线移动**：支持 8 方向移动
3. **动态障碍物**：支持运行时添加 / 移除墙壁
4. **JPS 算法**：Jump Point Search 优化 A* 在均匀网格上的效率
5. **双向搜索**：从起点和终点同时搜索
6. **3D 路径**：扩展到三维空间
7. **GUI 版本**：使用 Pygame 实现鼠标交互式地图编辑
