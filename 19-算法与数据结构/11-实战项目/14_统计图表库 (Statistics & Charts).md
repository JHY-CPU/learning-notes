# 统计图表工具 (Statistics & Charts)

## 项目需求与功能分析

数据分析离不开基本的统计计算和可视化。本项目用纯 Python 实现描述统计量计算和终端图表展示，无需第三方库。

### 核心功能

- 描述统计量（均值、中位数、众数、方差、标准差）
- 百分位数和四分位距
- 正态分布和均匀分布的随机数生成
- 终端柱状图、折线图、直方图
- 数据分布可视化
- 相关系数计算

## 核心算法原理

### 描述统计

- **均值** (Mean): 数据的算术平均
- **中位数** (Median): 排序后的中间值
- **众数** (Mode): 出现频率最高的值
- **方差** (Variance): 数据离散程度的度量
- **标准差** (Standard Deviation): 方差的平方根

### 相关系数

皮尔逊相关系数衡量两个变量的线性相关程度，值域 [-1, 1]。

## 完整代码实现

```python
import math
import random
from collections import Counter
from typing import List, Tuple, Optional


class Statistics:
    """描述统计计算器"""

    @staticmethod
    def mean(data: List[float]) -> float:
        return sum(data) / len(data) if data else 0

    @staticmethod
    def median(data: List[float]) -> float:
        s = sorted(data); n = len(s)
        if n == 0: return 0
        if n % 2 == 1: return s[n // 2]
        return (s[n // 2 - 1] + s[n // 2]) / 2

    @staticmethod
    def mode(data: List[float]):
        if not data: return None
        counter = Counter(data)
        max_freq = max(counter.values())
        modes = [k for k, v in counter.items() if v == max_freq]
        return modes[0] if len(modes) == 1 else modes

    @staticmethod
    def variance(data: List[float]) -> float:
        n = len(data)
        if n < 2: return 0
        m = Statistics.mean(data)
        return sum((x - m)**2 for x in data) / (n - 1)

    @staticmethod
    def std(data: List[float]) -> float:
        return math.sqrt(Statistics.variance(data))

    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        s = sorted(data)
        k = (len(s) - 1) * p / 100
        f = math.floor(k); c = math.ceil(k)
        if f == c: return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    @staticmethod
    def iqr(data: List[float]) -> float:
        return Statistics.percentile(data, 75) - Statistics.percentile(data, 25)

    @staticmethod
    def correlation(x: List[float], y: List[float]) -> float:
        n = len(x)
        if n != len(y) or n < 2: return 0
        mx, my = Statistics.mean(x), Statistics.mean(y)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
        dx = math.sqrt(sum((xi - mx)**2 for xi in x))
        dy = math.sqrt(sum((yi - my)**2 for yi in y))
        return num / (dx * dy) if dx * dy > 0 else 0

    @staticmethod
    def summary(data: List[float]) -> dict:
        s = sorted(data)
        return {
            'count': len(data),
            'mean': Statistics.mean(data),
            'median': Statistics.median(data),
            'std': Statistics.std(data),
            'min': s[0] if s else 0,
            'max': s[-1] if s else 0,
            'Q1': Statistics.percentile(data, 25),
            'Q3': Statistics.percentile(data, 75),
            'IQR': Statistics.iqr(data),
        }


class Distributions:
    """概率分布"""

    @staticmethod
    def uniform(n: int, a: float = 0, b: float = 1) -> List[float]:
        return [random.uniform(a, b) for _ in range(n)]

    @staticmethod
    def normal(n: int, mu: float = 0, sigma: float = 1) -> List[float]:
        """Box-Muller 正态分布"""
        result = []
        for _ in range((n + 1) // 2):
            u1 = random.random(); u2 = random.random()
            z1 = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            z2 = math.sqrt(-2 * math.log(u1)) * math.sin(2 * math.pi * u2)
            result.extend([mu + sigma * z1, mu + sigma * z2])
        return result[:n]

    @staticmethod
    def exponential(n: int, lam: float = 1.0) -> List[float]:
        return [-math.log(1 - random.random()) / lam for _ in range(n)]


class TerminalChart:
    """终端图表"""

    COLORS = {
        'red': '\033[91m', 'green': '\033[92m', 'yellow': '\033[93m',
        'blue': '\033[94m', 'cyan': '\033[96m', 'reset': '\033[0m',
    }

    @staticmethod
    def bar_chart(labels: List[str], values: List[float], title: str = "",
                  width: int = 40, color: str = 'blue'):
        """柱状图"""
        if not values: return
        max_val = max(values)
        c = TerminalChart.COLORS.get(color, '')
        rst = TerminalChart.COLORS['reset']

        print(f"\n  {title}")
        max_label = max(len(str(l)) for l in labels)
        for label, val in zip(labels, values):
            bar_len = int(val / max_val * width) if max_val > 0 else 0
            bar = c + '█' * bar_len + rst
            print(f"  {str(label):>{max_label}} | {bar} {val:.2f}")

    @staticmethod
    def line_chart(data: List[float], title: str = "", width: int = 60, height: int = 20):
        """折线图"""
        if not data: return
        mn, mx = min(data), max(data)
        rng = mx - mn if mx != mn else 1

        print(f"\n  {title}")
        grid = [[' '] * width for _ in range(height)]

        # 绘制数据点
        for i, val in enumerate(data):
            x = int(i / max(len(data) - 1, 1) * (width - 1))
            y = int((1 - (val - mn) / rng) * (height - 1))
            y = max(0, min(height - 1, y))
            grid[y][x] = '●'

        # Y 轴标签
        for r in range(height):
            y_val = mx - r * rng / max(height - 1, 1)
            label = f"{y_val:>8.2f} "
            print(f"  {label}│{''.join(grid[r])}")

        print(f"  {'':>9}└{'─' * width}")
        print(f"  {'':>10}0{' ' * (width - 8)}{len(data)-1}")

    @staticmethod
    def histogram(data: List[float], bins: int = 10, title: str = "",
                  width: int = 40):
        """直方图"""
        if not data: return
        mn, mx = min(data), max(data)
        rng = mx - mn if mx != mn else 1
        bin_width = rng / bins

        # 分箱
        counts = [0] * bins
        for val in data:
            idx = min(int((val - mn) / bin_width), bins - 1)
            counts[idx] += 1

        max_count = max(counts) if counts else 1
        print(f"\n  {title}")
        for i, count in enumerate(counts):
            lo = mn + i * bin_width
            hi = lo + bin_width
            bar_len = int(count / max_count * width)
            bar = '█' * bar_len
            print(f"  [{lo:>7.2f},{hi:>7.2f}) | {bar} {count}")

    @staticmethod
    def scatter_plot(x: List[float], y: List[float], title: str = "",
                     width: int = 50, height: int = 20):
        """散点图"""
        if not x or not y: return
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        x_rng = x_max - x_min or 1
        y_rng = y_max - y_min or 1

        grid = [[' '] * width for _ in range(height)]
        for xi, yi in zip(x, y):
            col = int((xi - x_min) / x_rng * (width - 1))
            row = int((1 - (yi - y_min) / y_rng) * (height - 1))
            col = max(0, min(width-1, col))
            row = max(0, min(height-1, row))
            grid[row][col] = '●'

        print(f"\n  {title}")
        for r in range(height):
            y_val = y_max - r * y_rng / max(height-1, 1)
            print(f"  {y_val:>8.2f} │{''.join(grid[r])}")
        print(f"  {'':>9}└{'─' * width}")


def demo():
    random.seed(42)
    data = Distributions.normal(1000, 50, 10)

    s = Statistics.summary(data)
    print("数据摘要:")
    for k, v in s.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    TerminalChart.histogram(data, bins=15, title="正态分布直方图 (mu=50, sigma=10)")

    # 柱状图
    TerminalChart.bar_chart(
        ['Python', 'Java', 'C++', 'Go', 'Rust'],
        [35, 25, 15, 12, 8],
        title="编程语言流行度"
    )


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestStatistics(unittest.TestCase):
    def test_mean_median(self):
        self.assertEqual(Statistics.mean([1,2,3,4,5]), 3.0)
        self.assertEqual(Statistics.median([1,3,5]), 3.0)
        self.assertEqual(Statistics.median([1,2,3,4]), 2.5)

    def test_variance_std(self):
        data = [2,4,4,4,5,5,7,9]
        self.assertAlmostEqual(Statistics.variance(data), 4.571, places=2)
        self.assertAlmostEqual(Statistics.std(data), 2.138, places=2)

    def test_correlation(self):
        x = [1,2,3,4,5]
        y = [2,4,6,8,10]
        self.assertAlmostEqual(Statistics.correlation(x, y), 1.0)

    def test_normal_distribution(self):
        data = Distributions.normal(10000, 0, 1)
        self.assertAlmostEqual(Statistics.mean(data), 0, places=1)
        self.assertAlmostEqual(Statistics.std(data), 1, places=1)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **箱线图**：终端 ASCII 箱线图
2. **回归分析**：线性回归拟合
3. **假设检验**：t 检验、卡方检验
4. **时间序列**：移动平均、趋势分析
5. **CSV 读取**：从文件加载数据
6. **HTML 导出**：生成 HTML 图表（结合 Chart.js）
7. **实时数据**：支持动态数据流的实时图表
