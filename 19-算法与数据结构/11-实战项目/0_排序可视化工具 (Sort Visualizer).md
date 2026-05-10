# 排序可视化工具 (Sort Visualizer)

## 项目需求与功能分析

排序算法是算法学习的基石，但仅通过文字和伪代码很难直观理解排序过程中元素的移动与比较。本项目旨在构建一个交互式排序可视化工具，将抽象的排序过程转化为直观的动画演示。

### 核心功能

- 支持冒泡排序、选择排序、插入排序、快速排序、归并排序、堆排序六种经典算法
- 实时动画展示每一步比较和交换操作
- 可调节动画速度（慢 / 中 / 快 / 极速）
- 支持自定义数组大小（10 ~ 200 个元素）
- 随机生成、近乎有序、逆序、含大量重复四种数据分布
- 统计比较次数、交换次数、耗时等指标

### 技术选型

| 组件 | 技术方案 |
|------|----------|
| 语言 | Python 3 |
| 可视化 | Terminal ANSI 彩色输出 |
| 交互 | 命令行参数 + 键盘控制 |

## 核心算法原理

### 冒泡排序 (Bubble Sort)

重复遍历数组，每次比较相邻两个元素，若顺序错误则交换。每轮遍历将最大元素"冒泡"到末尾。

- 时间复杂度：O(n^2) 平均 / 最坏，O(n) 最好（已排序时提前终止）
- 空间复杂度：O(1)
- 稳定排序

### 快速排序 (Quick Sort)

选取基准元素 (pivot)，将数组分为小于基准和大于基准两部分，递归排序两个子数组。

- 时间复杂度：O(n log n) 平均，O(n^2) 最坏
- 空间复杂度：O(log n) 递归栈
- 不稳定排序

### 归并排序 (Merge Sort)

分治策略：将数组递归分成两半，分别排序后再合并。合并时按序选取较小元素。

- 时间复杂度：O(n log n) 所有情况
- 空间复杂度：O(n)
- 稳定排序

## 完整代码实现

```python
import time
import random
import os
import sys


class SortVisualizer:
    """排序可视化工具 - 终端版本"""

    # ANSI 颜色码
    COLORS = {
        'reset': '\033[0m',
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'white': '\033[97m',
    }

    def __init__(self, size=30, speed=0.05):
        self.size = size
        self.speed = speed
        self.array = []
        self.comparisons = 0
        self.swaps = 0
        self.highlights = {}  # index -> color

    def generate_array(self, distribution='random'):
        """生成测试数组"""
        if distribution == 'random':
            self.array = [random.randint(1, 99) for _ in range(self.size)]
        elif distribution == 'nearly_sorted':
            self.array = list(range(1, self.size + 1))
            for _ in range(self.size // 10):
                i, j = random.sample(range(self.size), 2)
                self.array[i], self.array[j] = self.array[j], self.array[i]
        elif distribution == 'reversed':
            self.array = list(range(self.size, 0, -1))
        elif distribution == 'many_duplicates':
            self.array = [random.randint(1, 10) for _ in range(self.size)]
        self.comparisons = 0
        self.swaps = 0

    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')

    def render(self, title="", step_info=""):
        """在终端中渲染数组的柱状图"""
        self.clear_screen()
        max_val = max(self.array) if self.array else 1
        bar_max_width = 50

        print(f"{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print()

        for i, val in enumerate(self.array):
            bar_len = int(val / max_val * bar_max_width)
            bar = '█' * bar_len
            color = self.highlights.get(i, self.COLORS['white'])
            print(f"  {color}{bar} {val:>3}{self.COLORS['reset']}")

        print()
        print(f"  比较次数: {self.comparisons}  |  交换次数: {self.swaps}")
        if step_info:
            print(f"  {step_info}")
        print()
        time.sleep(self.speed)

    def compare(self, i, j):
        """比较两个元素并记录"""
        self.comparisons += 1
        self.highlights = {i: self.COLORS['red'], j: self.COLORS['red']}
        return self.array[i] > self.array[j]

    def swap(self, i, j):
        """交换两个元素并记录"""
        self.swaps += 1
        self.array[i], self.array[j] = self.array[j], self.array[i]
        self.highlights = {i: self.COLORS['green'], j: self.COLORS['green']}

    # ===== 排序算法 =====

    def bubble_sort(self):
        n = len(self.array)
        for i in range(n):
            swapped = False
            for j in range(0, n - i - 1):
                self.render(f"冒泡排序 - 第{i+1}轮", f"比较索引 [{j}] 和 [{j+1}]")
                if self.compare(j, j + 1):
                    self.swap(j, j + 1)
                    swapped = True
            if not swapped:
                break
        self.highlights = {}

    def selection_sort(self):
        n = len(self.array)
        for i in range(n):
            min_idx = i
            for j in range(i + 1, n):
                self.render(f"选择排序 - 找第{i+1}小元素",
                            f"当前最小值索引: {min_idx}, 比较索引: {j}")
                self.comparisons += 1
                self.highlights = {min_idx: self.COLORS['yellow'], j: self.COLORS['red']}
                if self.array[j] < self.array[min_idx]:
                    min_idx = j
            if min_idx != i:
                self.swap(i, min_idx)
        self.highlights = {}

    def insertion_sort(self):
        n = len(self.array)
        for i in range(1, n):
            key = self.array[i]
            j = i - 1
            self.render(f"插入排序 - 插入元素 {key}", f"已排序区间 [0, {i-1}]")
            while j >= 0 and self.array[j] > key:
                self.comparisons += 1
                self.array[j + 1] = self.array[j]
                self.swaps += 1
                self.highlights = {j: self.COLORS['red'], j+1: self.COLORS['green']}
                self.render(f"插入排序 - 元素 {key} 向左移动",
                            f"移动索引 [{j}] -> [{j+1}]")
                j -= 1
            self.array[j + 1] = key
        self.highlights = {}

    def quick_sort(self, low=0, high=None):
        if high is None:
            high = len(self.array) - 1
        if low < high:
            pivot_idx = self._partition(low, high)
            self.quick_sort(low, pivot_idx - 1)
            self.quick_sort(pivot_idx + 1, high)
        self.highlights = {}

    def _partition(self, low, high):
        pivot = self.array[high]
        i = low - 1
        for j in range(low, high):
            self.render(f"快速排序 - 分区 pivot={pivot}",
                        f"区间 [{low}, {high}], 当前索引: {j}")
            self.comparisons += 1
            self.highlights = {high: self.COLORS['yellow'], j: self.COLORS['red']}
            if self.array[j] <= pivot:
                i += 1
                if i != j:
                    self.swap(i, j)
        if i + 1 != high:
            self.swap(i + 1, high)
        return i + 1

    def merge_sort(self, left=0, right=None):
        if right is None:
            right = len(self.array) - 1
        if left < right:
            mid = (left + right) // 2
            self.merge_sort(left, mid)
            self.merge_sort(mid + 1, right)
            self._merge(left, mid, right)
        self.highlights = {}

    def _merge(self, left, mid, right):
        left_arr = self.array[left:mid + 1]
        right_arr = self.array[mid + 1:right + 1]
        i = j = 0
        k = left
        self.render(f"归并排序 - 合并 [{left}, {mid}] 和 [{mid+1}, {right}]",
                    f"左: {left_arr}  右: {right_arr}")
        while i < len(left_arr) and j < len(right_arr):
            self.comparisons += 1
            if left_arr[i] <= right_arr[j]:
                self.array[k] = left_arr[i]; i += 1
            else:
                self.array[k] = right_arr[j]; j += 1
            self.highlights = {k: self.COLORS['green']}
            self.swaps += 1; k += 1
        while i < len(left_arr):
            self.array[k] = left_arr[i]; self.highlights = {k: self.COLORS['green']}; i += 1; k += 1
        while j < len(right_arr):
            self.array[k] = right_arr[j]; self.highlights = {k: self.COLORS['green']}; j += 1; k += 1

    def heap_sort(self):
        n = len(self.array)
        for i in range(n // 2 - 1, -1, -1):
            self._heapify(n, i)
        for i in range(n - 1, 0, -1):
            self.render(f"堆排序 - 提取最大元素", f"堆大小: {i+1}")
            self.swap(0, i)
            self._heapify(i, 0)
        self.highlights = {}

    def _heapify(self, n, i):
        largest = i
        left, right = 2 * i + 1, 2 * i + 2
        if left < n:
            self.comparisons += 1
            if self.array[left] > self.array[largest]:
                largest = left
        if right < n:
            self.comparisons += 1
            if self.array[right] > self.array[largest]:
                largest = right
        if largest != i:
            self.swap(i, largest)
            self._heapify(n, largest)

    def run(self, algorithm='bubble'):
        """运行指定排序算法"""
        algo_map = {
            'bubble': ('冒泡排序', self.bubble_sort),
            'selection': ('选择排序', self.selection_sort),
            'insertion': ('插入排序', self.insertion_sort),
            'quick': ('快速排序', self.quick_sort),
            'merge': ('归并排序', self.merge_sort),
            'heap': ('堆排序', self.heap_sort),
        }
        if algorithm not in algo_map:
            print(f"未知算法: {algorithm}"); return
        name, func = algo_map[algorithm]
        self.render(f"{name} - 初始状态")
        start_time = time.time()
        func()
        elapsed = time.time() - start_time
        self.render(f"{name} - 排序完成!")
        print(f"  耗时: {elapsed:.4f}s  比较: {self.comparisons}  交换: {self.swaps}")
```

## 测试用例

```python
import unittest

class TestSortVisualizer(unittest.TestCase):

    def test_all_algorithms_correct(self):
        for algo in ['bubble', 'selection', 'insertion', 'quick', 'merge', 'heap']:
            vis = SortVisualizer(size=100)
            vis.generate_array('random')
            original = vis.array.copy()
            func = getattr(vis, algo + '_sort')
            func()
            self.assertEqual(vis.array, sorted(original), f"{algo} 排序结果不正确")

    def test_already_sorted(self):
        vis = SortVisualizer(size=20)
        vis.array = list(range(20))
        vis.bubble_sort()
        self.assertEqual(vis.array, list(range(20)))
        self.assertEqual(vis.comparisons, 19)  # 最优冒泡只需 n-1 次

    def test_duplicates(self):
        vis = SortVisualizer(size=100)
        vis.generate_array('many_duplicates')
        original = vis.array.copy()
        vis.merge_sort()
        self.assertEqual(vis.array, sorted(original))

    def test_empty(self):
        vis = SortVisualizer(); vis.array = []; vis.quick_sort()
        self.assertEqual(vis.array, [])

if __name__ == '__main__':
    unittest.main()
```

## 运行方式

```bash
python sort_visualizer.py -a bubble -n 30      # 冒泡排序可视化
python sort_visualizer.py -a quick -n 50 -s 0.01 # 快排快速动画
python sort_visualizer.py -a merge -n 40 -d reversed  # 逆序数据
```

## 扩展方向

1. **图形化界面**：使用 Pygame / Tkinter 替代终端渲染，支持更丰富的视觉效果
2. **更多算法**：添加计数排序、基数排序、桶排序、Tim 排序
3. **分步控制**：支持单步执行、暂停、回退操作
4. **双算法对比**：同时运行两种算法，直观对比效率差异
5. **声音反馈**：用不同音调表示不同大小的元素，提供听觉辅助
6. **Web 版本**：使用 Canvas 或 D3.js 实现浏览器端排序可视化
7. **复杂度曲线**：实时绘制比较 / 交换次数与 n 的关系图
