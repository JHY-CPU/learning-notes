# 迷宫生成与求解 (Maze Generator & Solver)

## 项目需求与功能分析

迷宫是算法可视化的经典载体。本项目实现迷宫的自动生成（DFS 递归回溯）和自动求解（BFS 最短路径），并提供终端动画展示。

### 核心功能

- DFS 递归回溯法生成完美迷宫（保证连通且无环）
- 随机 Prim 算法生成迷宫
- BFS 求解最短路径
- DFS 求解任意路径
- 终端动画展示生成与求解过程
- 迷宫尺寸自定义

### 迷宫表示

使用扩展网格，将每个单元格扩展为 2x2 的格子，墙壁占据奇数行列：

```
# # # # #     # = 墙壁
# . # . #     . = 通路
# # # # #
# . . . #
# # # # #
```

## 核心算法原理

### DFS 递归回溯生成

1. 从起点开始，标记为已访问
2. 随机选择一个未访问的相邻单元格
3. 打通当前单元格与相邻单元格之间的墙壁
4. 递归进入相邻单元格
5. 若无未访问邻居，回溯

生成的迷宫是完美迷宫（任意两点之间有且仅有一条路径）。

### BFS 最短路径求解

1. 从起点开始，将起点加入队列
2. 取出队首节点，将未访问的邻居加入队列并记录前驱
3. 重复直到到达终点
4. 从前驱字典回溯得到最短路径

## 完整代码实现

```python
import random, time, os
from collections import deque
from typing import List, Tuple, Optional, Dict


class Maze:
    WALL = '#'
    PATH = ' '
    START = 'S'
    END = 'E'
    VISITED = '.'
    SOLUTION = '*'

    def __init__(self, width=20, height=20):
        self.width = width
        self.height = height
        # 扩展网格: (2*height+1) x (2*width+1)
        self.ew = 2 * width + 1
        self.eh = 2 * height + 1
        self.grid = [[self.WALL] * self.ew for _ in range(self.eh)]
        self.start = (1, 1)
        self.end = (self.eh - 2, self.ew - 2)

    def _cell_to_grid(self, r, c):
        """将迷宫单元格坐标转换为扩展网格坐标"""
        return (2 * r + 1, 2 * c + 1)

    def _neighbors(self, r, c):
        """获取迷宫单元格的相邻单元格"""
        result = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.height and 0 <= nc < self.width:
                result.append((nr, nc))
        return result

    def generate_dfs(self):
        """DFS 递归回溯生成迷宫"""
        visited = set()
        stack = [(0, 0)]
        visited.add((0, 0))
        gr, gc = self._cell_to_grid(0, 0)
        self.grid[gr][gc] = self.PATH

        while stack:
            r, c = stack[-1]
            unvisited = [
                (nr, nc) for nr, nc in self._neighbors(r, c)
                if (nr, nc) not in visited
            ]
            if unvisited:
                nr, nc = random.choice(unvisited)
                # 打通墙壁
                wall_r = r + nr + 1
                wall_c = c + nc + 1
                self.grid[wall_r][wall_c] = self.PATH
                gr, gc = self._cell_to_grid(nr, nc)
                self.grid[gr][gc] = self.PATH
                visited.add((nr, nc))
                stack.append((nr, nc))
            else:
                stack.pop()

        # 标记起点终点
        sr, sc = self._cell_to_grid(0, 0)
        er, ec = self._cell_to_grid(self.height - 1, self.width - 1)
        self.grid[sr][sc] = self.START
        self.grid[er][ec] = self.END
        self.start = (sr, sc)
        self.end = (er, ec)

    def generate_prim(self):
        """随机 Prim 算法生成迷宫"""
        visited = set()
        walls = []

        # 从 (0,0) 开始
        visited.add((0, 0))
        gr, gc = self._cell_to_grid(0, 0)
        self.grid[gr][gc] = self.PATH

        # 添加邻居墙壁
        for nr, nc in self._neighbors(0, 0):
            walls.append((0, 0, nr, nc))

        while walls:
            idx = random.randint(0, len(walls) - 1)
            r1, c1, r2, c2 = walls[idx]
            walls[idx] = walls[-1]
            walls.pop()

            if (r2, c2) in visited:
                continue

            # 打通
            wall_r = r1 + r2 + 1
            wall_c = c1 + c2 + 1
            self.grid[wall_r][wall_c] = self.PATH
            gr, gc = self._cell_to_grid(r2, c2)
            self.grid[gr][gc] = self.PATH
            visited.add((r2, c2))

            for nr, nc in self._neighbors(r2, c2):
                if (nr, nc) not in visited:
                    walls.append((r2, c2, nr, nc))

        sr, sc = self._cell_to_grid(0, 0)
        er, ec = self._cell_to_grid(self.height - 1, self.width - 1)
        self.grid[sr][sc] = self.START
        self.grid[er][ec] = self.END
        self.start = (sr, sc)
        self.end = (er, ec)

    def solve_bfs(self, animate=True, speed=0.02):
        """BFS 求解最短路径"""
        queue = deque([self.start])
        parent: Dict = {self.start: None}
        visited = {self.start}

        while queue:
            cur = queue.popleft()
            if cur == self.end:
                return self._draw_path(parent, animate)

            r, c = cur
            if cur != self.start:
                self.grid[r][c] = self.VISITED

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.eh and 0 <= nc < self.ew
                        and (nr, nc) not in visited
                        and self.grid[nr][nc] != self.WALL):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = cur
                    queue.append((nr, nc))

            if animate:
                self.render("BFS 求解中...")
                time.sleep(speed)

        return None

    def solve_dfs(self, animate=True, speed=0.02):
        """DFS 求解路径（不一定最短）"""
        stack = [self.start]
        parent: Dict = {self.start: None}
        visited = {self.start}

        while stack:
            cur = stack.pop()
            if cur == self.end:
                return self._draw_path(parent, animate)

            r, c = cur
            if cur != self.start:
                self.grid[r][c] = self.VISITED

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.eh and 0 <= nc < self.ew
                        and (nr, nc) not in visited
                        and self.grid[nr][nc] != self.WALL):
                    visited.add((nr, nc))
                    parent[(nr, nc)] = cur
                    stack.append((nr, nc))

            if animate:
                self.render("DFS 求解中...")
                time.sleep(speed)

        return None

    def _draw_path(self, parent, animate):
        path = []
        node = self.end
        while node is not None:
            path.append(node)
            node = parent[node]
        path.reverse()

        for r, c in path:
            if (r, c) not in (self.start, self.end):
                self.grid[r][c] = self.SOLUTION
            if animate:
                self.render("回溯路径...")
                time.sleep(0.01)

        return path

    def render(self, title=""):
        os.system('cls' if os.name == 'nt' else 'clear')
        colors = {
            self.WALL: '\033[90m##\033[0m',
            self.PATH: '  ',
            self.START: '\033[92mSS\033[0m',
            self.END: '\033[91mEE\033[0m',
            self.VISITED: '\033[94m..\033[0m',
            self.SOLUTION: '\033[93m**\033[0m',
        }
        print(f"  {title}")
        for row in self.grid:
            print('  ' + ''.join(colors.get(c, c*2) for c in row))


def demo():
    print("生成迷宫...")
    maze = Maze(width=15, height=15)
    maze.generate_dfs()
    maze.render("迷宫已生成 (DFS)")

    print("\n按 Enter 开始 BFS 求解...")
    input()
    path = maze.solve_bfs(animate=True, speed=0.01)
    if path:
        print(f"\n最短路径长度: {len(path)}")
    else:
        print("\n无解!")


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestMaze(unittest.TestCase):
    def test_dfs_generates_connected(self):
        m = Maze(10, 10); m.generate_dfs()
        path = m.solve_bfs(animate=False)
        self.assertIsNotNone(path)

    def test_prim_generates_connected(self):
        m = Maze(10, 10); m.generate_prim()
        path = m.solve_bfs(animate=False)
        self.assertIsNotNone(path)

    def test_bfs_shorter_than_dfs(self):
        m = Maze(20, 20); m.generate_dfs()
        import copy
        m2 = Maze(20, 20); m2.grid = copy.deepcopy(m.grid)
        m2.start, m2.end = m.start, m.end
        bfs_path = m.solve_bfs(animate=False)
        dfs_path = m2.solve_dfs(animate=False)
        self.assertLessEqual(len(bfs_path), len(dfs_path))

    def test_small_maze(self):
        m = Maze(2, 2); m.generate_dfs()
        path = m.solve_bfs(animate=False)
        self.assertIsNotNone(path)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **加权迷宫**：通路带有不同通行代价
2. **环形迷宫**：允许在迷宫中形成环路
3. **多层迷宫**：3D 迷宫，包含楼梯连接不同层
4. **迷宫竞赛**：两个算法同时求解，可视化对比
5. **自动生成题目**：控制迷宫难度（路径长度、分岔数）
6. **ASCII 导出**：将迷宫保存为文本文件
7. **Pygame 版本**：图形化交互式迷宫编辑与求解
