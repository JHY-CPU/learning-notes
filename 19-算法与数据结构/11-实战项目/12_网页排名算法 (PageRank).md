# 网页排名算法 (PageRank)

## 项目需求与功能分析

PageRank 是 Google 搜索引擎的核心算法之一，通过分析网页之间的链接关系来评估网页的重要性。本项目实现 PageRank 的迭代计算，并支持多种变体。

### 核心功能

- 网页链接图构建
- PageRank 迭代计算（幂迭代法）
- 阻尼因子处理悬挂节点
- 支持 Topic-Sensitive PageRank
- 排名结果可视化
- 收敛过程分析

### 算法思想

一个网页的重要性由链接到它的网页数量和质量决定。重要网页的链接更有价值。

## 核心算法原理

### 经典 PageRank 公式

```
PR(i) = (1-d)/N + d * sum(PR(j) / L(j))   for all j -> i
```

- d: 阻尼因子（通常 0.85），表示用户继续点击链接的概率
- N: 网页总数
- L(j): 网页 j 的出链数

### 幂迭代法

1. 初始化所有网页 PR 值为 1/N
2. 反复应用公式更新 PR 值
3. 当 PR 值变化小于阈值时收敛

## 完整代码实现

```python
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import random


class WebGraph:
    """网页链接图"""

    def __init__(self):
        self.outlinks: Dict[str, Set[str]] = defaultdict(set)
        self.inlinks: Dict[str, Set[str]] = defaultdict(set)
        self.pages: Set[str] = set()

    def add_link(self, from_page: str, to_page: str):
        self.pages.add(from_page)
        self.pages.add(to_page)
        self.outlinks[from_page].add(to_page)
        self.inlinks[to_page].add(from_page)

    def add_page(self, page: str):
        self.pages.add(page)

    def out_degree(self, page: str) -> int:
        return len(self.outlinks[page])

    def dangling_pages(self) -> List[str]:
        """无出链的页面"""
        return [p for p in self.pages if self.out_degree(p) == 0]


class PageRank:
    """PageRank 计算器"""

    def __init__(self, graph: WebGraph, damping=0.85, max_iter=100, tol=1e-6):
        self.graph = graph
        self.damping = damping
        self.max_iter = max_iter
        self.tol = tol
        self.ranks: Dict[str, float] = {}
        self.history: List[Dict[str, float]] = []

    def compute(self) -> Dict[str, float]:
        """幂迭代法计算 PageRank"""
        N = len(self.graph.pages)
        if N == 0:
            return {}

        # 初始化
        self.ranks = {page: 1.0 / N for page in self.graph.pages}
        self.history = [dict(self.ranks)]

        for iteration in range(self.max_iter):
            new_ranks = {}
            dangling_sum = sum(
                self.ranks[p] for p in self.graph.dangling_pages()
            )

            for page in self.graph.pages:
                # 来自入链的贡献
                link_sum = sum(
                    self.ranks[src] / self.graph.out_degree(src)
                    for src in self.graph.inlinks[page]
                    if self.graph.out_degree(src) > 0
                )
                # 悬挂节点均匀分配
                dangling_contribution = dangling_sum / N
                # PageRank 公式
                new_ranks[page] = (
                    (1 - self.damping) / N
                    + self.damping * (link_sum + dangling_contribution)
                )

            # 检查收敛
            diff = sum(abs(new_ranks[p] - self.ranks[p]) for p in self.graph.pages)
            self.ranks = new_ranks
            self.history.append(dict(self.ranks))

            if diff < self.tol:
                print(f"  收敛于第 {iteration + 1} 次迭代 (diff={diff:.8f})")
                break

        return self.ranks

    def compute_topic_sensitive(self, topic_pages: Set[str],
                                 teleport_prob: float = 0.1) -> Dict[str, float]:
        """Topic-Sensitive PageRank"""
        N = len(self.graph.pages)
        self.ranks = {page: 1.0 / N for page in self.graph.pages}

        for _ in range(self.max_iter):
            new_ranks = {}
            dangling_sum = sum(self.ranks[p] for p in self.graph.dangling_pages())

            for page in self.graph.pages:
                link_sum = sum(
                    self.ranks[src] / self.graph.out_degree(src)
                    for src in self.graph.inlinks[page]
                    if self.graph.out_degree(src) > 0
                )
                # 传送概率偏向主题相关页面
                teleport = teleport_prob / len(topic_pages) if page in topic_pages else 0
                new_ranks[page] = (
                    (1 - self.damping) * (1 - teleport_prob) / N
                    + (1 - self.damping) * teleport
                    + self.damping * (link_sum + dangling_sum / N)
                )

            diff = sum(abs(new_ranks[p] - self.ranks[p]) for p in self.graph.pages)
            self.ranks = new_ranks
            if diff < self.tol:
                break

        return self.ranks

    def get_top(self, n: int = 10) -> List[Tuple[str, float]]:
        """获取排名前 n 的页面"""
        sorted_ranks = sorted(self.ranks.items(), key=lambda x: x[1], reverse=True)
        return sorted_ranks[:n]

    def display_ranks(self):
        """显示排名"""
        top = self.get_top(len(self.ranks))
        print(f"\n{'排名':<6} {'页面':<25} {'PageRank':<12}")
        print("-" * 45)
        for i, (page, rank) in enumerate(top, 1):
            bar = '█' * int(rank * 200)
            print(f"{i:<6} {page:<25} {rank:<12.6f} {bar}")

    def display_convergence(self):
        """显示收敛过程"""
        if not self.history:
            return
        pages = list(self.ranks.keys())[:5]
        print(f"\n{'迭代':<6}", end='')
        for p in pages:
            print(f"{p[:10]:<12}", end='')
        print()
        for i, ranks in enumerate(self.history[:20]):
            print(f"{i:<6}", end='')
            for p in pages:
                print(f"{ranks.get(p,0):<12.6f}", end='')
            print()


def random_surfer_simulation(graph: WebGraph, damping=0.85,
                              steps=100000) -> Dict[str, float]:
    """随机游走模拟 PageRank"""
    pages = list(graph.pages)
    N = len(pages)
    counts = defaultdict(int)

    current = random.choice(pages)
    for _ in range(steps):
        counts[current] += 1
        if random.random() < damping and graph.out_degree(current) > 0:
            current = random.choice(list(graph.outlinks[current]))
        else:
            current = random.choice(pages)

    total = sum(counts.values())
    return {p: counts[p] / total for p in pages}


def demo():
    # 构建一个简单的网页图
    graph = WebGraph()
    links = [
        ('A', 'B'), ('A', 'C'),
        ('B', 'C'),
        ('C', 'A'),
        ('D', 'C'), ('D', 'B'),
        ('E', 'D'), ('E', 'B'),
        ('F', 'E'), ('F', 'B'),
    ]
    for f, t in links:
        graph.add_link(f, t)

    # PageRank 计算
    pr = PageRank(graph, damping=0.85)
    pr.compute()
    pr.display_ranks()

    # 随机游走验证
    print("\n随机游走模拟 (100000 步):")
    sim = random_surfer_simulation(graph, steps=100000)
    for page in sorted(sim, key=sim.get, reverse=True):
        print(f"  {page}: {sim[page]:.4f} (迭代: {pr.ranks[page]:.4f})")


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestPageRank(unittest.TestCase):
    def test_simple_graph(self):
        g = WebGraph()
        g.add_link('A','B'); g.add_link('B','C'); g.add_link('C','A')
        pr = PageRank(g); pr.compute()
        total = sum(pr.ranks.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    def test_dangling_page(self):
        g = WebGraph()
        g.add_link('A','B'); g.add_page('C')  # C 是悬挂节点
        pr = PageRank(g); pr.compute()
        self.assertIn('C', pr.ranks)
        self.assertGreater(pr.ranks['C'], 0)

    def test_convergence(self):
        g = WebGraph()
        for f,t in [('A','B'),('B','C'),('C','A'),('D','A')]: g.add_link(f,t)
        pr = PageRank(g, max_iter=100)
        pr.compute()
        self.assertGreater(len(pr.history), 1)

    def test_random_surfer(self):
        g = WebGraph()
        g.add_link('A','B'); g.add_link('B','A')
        sim = random_surfer_simulation(g, steps=50000)
        self.assertAlmostEqual(sim['A'] + sim['B'], 1.0, places=1)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **HITS 算法**：实现 Hub 和 Authority 的交替计算
2. **分布式计算**：使用 MapReduce 并行计算大规模图
3. **增量更新**：网页图变化时增量更新 PageRank
4. **Personalized PageRank**：个性化 PageRank
5. **链接分析**：检测链接农场和 SEO 作弊
6. **时间衰减**：给较新的链接更高权重
7. **大规模模拟**：使用稀疏矩阵优化大图计算
