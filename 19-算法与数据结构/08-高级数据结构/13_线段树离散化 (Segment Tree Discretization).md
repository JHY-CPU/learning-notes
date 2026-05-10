# 线段树与离散化 (Segment Tree with Discretization)

## 1. 概述

当线段树处理的值域很大（如 10^9）但实际操作的数量较少（如 10^5）时，直接建树会浪费大量空间。**离散化**（Discretization）或**坐标压缩**（Coordinate Compression）将大值域映射到紧凑的连续整数，从而大幅减小线段树的规模。

## 2. 什么是离散化？

离散化是将原始的、可能稀疏的值域映射到连续的 [0, m-1] 区间的过程，其中 m 是实际出现的不同值的个数。

### 2.1 示例

原始数据：[1, 1000000000, 5, 1000000000, 3]
离散化后：[0, 3, 2, 3, 1]（按排序后的位置映射）

原始值  ->  离散化值
1       ->  0
3       ->  1
5       ->  2
10^9    ->  3

## 3. 离散化步骤

```python
def discretize(values):
    """
    离散化：将值域压缩到 [0, m-1]
    返回映射后的数组和原始值到离散值的映射
    """
    # 1. 去重并排序
    sorted_unique = sorted(set(values))

    # 2. 建立映射
    rank = {val: i for i, val in enumerate(sorted_unique)}

    # 3. 替换
    compressed = [rank[v] for v in values]

    return compressed, sorted_unique, rank
```

### 3.1 完整示例

```python
def discretize_example():
    values = [100, 500, 200, 100, 300, 500, 150]
    compressed, sorted_vals, rank = discretize(values)

    print(f"原始值: {values}")
    print(f"离散化: {compressed}")
    print(f"排序去重: {sorted_vals}")
    print(f"映射表: {rank}")
    # 原始值: [100, 500, 200, 100, 300, 500, 150]
    # 离散化: [0, 4, 2, 0, 3, 4, 1]
    # 排序去重: [100, 150, 200, 300, 500]
```

## 4. 离散化 + 线段树完整实现

### 4.1 问题描述

给定 n 个操作，每个操作是对值域 [1, 10^9] 中某个位置的修改或查询。n <= 10^5。

### 4.2 完整代码

```python
class DiscretizedSegmentTree:
    """离散化线段树"""

    def __init__(self, values):
        # 收集所有出现的值并离散化
        self.sorted_vals = sorted(set(values))
        self.rank = {v: i for i, v in enumerate(self.sorted_vals)}
        self.m = len(self.sorted_vals)

        # 建立线段树
        self.tree = [0] * (4 * self.m)
        self.lazy = [0] * (4 * self.m)

    def _build(self, node, l, r):
        if l == r:
            self.tree[node] = 0
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid)
        self._build(2 * node + 1, mid + 1, r)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, val, delta):
        """更新原始值 val 对应的位置"""
        idx = self.rank[val]
        self._update(1, 0, self.m - 1, idx, delta)

    def _update(self, node, l, r, idx, delta):
        if l == r:
            self.tree[node] += delta
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, delta)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, delta)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, val_l, val_r):
        """查询原始值 [val_l, val_r] 范围内的和"""
        idx_l = self.rank[val_l]
        idx_r = self.rank[val_r]
        return self._query(1, 0, self.m - 1, idx_l, idx_r)

    def _query(self, node, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.tree[node]
        if r < ql or l > qr:
            return 0
        mid = (l + r) // 2
        return (self._query(2 * node, l, mid, ql, qr) +
                self._query(2 * node + 1, mid + 1, r, qr))
```

## 5. 处理区间操作的离散化

当操作涉及区间 [l, r] 时，需要特殊处理：

### 5.1 方法1：扩展端点

对于每个区间 [l, r]，将 l, l-1, r, r+1 都加入离散化集合。

```python
def discretize_intervals(intervals):
    """离散化区间端点"""
    points = set()
    for l, r in intervals:
        points.add(l)
        points.add(l - 1)  # 处理左边界
        points.add(r)
        points.add(r + 1)  # 处理右边界

    sorted_points = sorted(points)
    rank = {v: i for i, v in enumerate(sorted_points)}

    return sorted_points, rank
```

### 5.2 方法2：使用相邻点间的虚拟节点

对于排序后的相邻点 a[i] 和 a[i+1]，如果它们之间有空隙，插入一个虚拟节点表示中间区域。

## 6. 使用 bisect 进行离散化

```python
from bisect import bisect_left

def rank_of(val, sorted_vals):
    """用二分查找获取离散化值"""
    return bisect_left(sorted_vals, val)

def discretize_with_bisect(values):
    """使用bisect的离散化"""
    sorted_vals = sorted(set(values))
    compressed = [bisect_left(sorted_vals, v) for v in values]
    return compressed, sorted_vals
```

## 7. C++ 离散化实现

```cpp
#include <vector>
#include <algorithm>
#include <map>
using namespace std;

vector<int> discretize(vector<int>& values) {
    // 复制并去重排序
    vector<int> sorted_vals = values;
    sort(sorted_vals.begin(), sorted_vals.end());
    sorted_vals.erase(unique(sorted_vals.begin(), sorted_vals.end()),
                      sorted_vals.end());

    // 建立映射
    map<int, int> rank;
    for (int i = 0; i < (int)sorted_vals.size(); i++) {
        rank[sorted_vals[i]] = i;
    }

    // 替换
    vector<int> result;
    for (int v : values) {
        result.push_back(rank[v]);
    }

    return result;
}

// 使用lower_bound的快速版本
int getRank(vector<int>& sorted_vals, int val) {
    return lower_bound(sorted_vals.begin(), sorted_vals.end(), val)
           - sorted_vals.begin();
}
```

## 8. 空间优化效果

| 原始值域 | 操作次数 | 离散化后空间 | 节省 |
|----------|---------|-------------|------|
| 10^9 | 10^5 | 4 * 10^5 | 99.96% |
| 10^9 | 10^4 | 4 * 10^4 | 99.996% |
| 10^18 | 10^5 | 4 * 10^5 | ~100% |

## 9. 注意事项

1. **值域变化**：如果操作中引入新值，需要动态扩展离散化
2. **区间查询**：注意离散化后区间端点的对应关系
3. **重复值**：离散化前必须去重
4. **稳定排序**：离散化保持原始值的相对大小关系

## 10. 总结

离散化是处理大值域线段树问题的关键技术：
- 将 O(V) 空间优化为 O(n)，其中 V >> n
- 时间复杂度不变，只是将原始值映射为离散值
- 适用于值域大但操作少的场景
- 区间操作需要注意端点的处理
