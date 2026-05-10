# 珂朵莉树 (Chtholly Tree / ODT)

## 1. 概述

珂朵莉树（Chtholly Tree），又称ODT（Old Driver Tree），是一种利用**有序集合（set）维护连续相同值区间**的数据结构。它在**数据随机**的前提下效率极高，被广泛用于算法竞赛。

起源：来自 Codeforces 上用户 Chtholly 的一个著名解法。

## 2. 核心思想

将数组中**值相同的连续段**合并为一个区间节点，用 set 维护。区间推平操作（将 [l, r] 全部赋值为 val）可以大幅减少区间数量。

## 3. 数据结构

### 3.1 区间节点

```python
from sortedcontainers import SortedSet
# 或者使用 Python 内置的 set + bisect

class Node:
    """珂朵莉树节点：表示一个值相同的连续区间"""
    def __init__(self, l, r, val):
        self.l = l      # 左端点
        self.r = r      # 右端点
        self.val = val  # 区间内所有元素的值

    def __lt__(self, other):
        return self.l < other.l

    def __repr__(self):
        return f"[{self.l},{self.r}]={self.val}"
```

### 3.2 使用 sorted list 实现

```python
class ChthollyTree:
    """珂朵莉树"""

    def __init__(self, data):
        """
        从数组初始化
        data: 初始数组
        """
        self.tree = SortedList()  # 需要 sortedcontainers 库
        n = len(data)
        i = 0
        while i < n:
            # 合并连续相同值
            j = i
            while j < n and data[j] == data[i]:
                j += 1
            self.tree.add(Node(i, j - 1, data[i]))
            i = j
```

### 3.3 使用内置 set 实现（竞赛常用）

```python
class ChthollyTreeSimple:
    """珂朵莉树（使用Python内置list+二分）"""

    def __init__(self, data):
        self.nodes = []  # 按左端点排序的区间列表
        n = len(data)
        i = 0
        while i < n:
            j = i
            while j < n and data[j] == data[i]:
                j += 1
            self.nodes.append(Node(i, j - 1, data[i]))
            i = j

    def _find(self, pos):
        """找到包含位置pos的区间"""
        lo, hi = 0, len(self.nodes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.nodes[mid].r < pos:
                lo = mid + 1
            elif self.nodes[mid].l > pos:
                hi = mid - 1
            else:
                return mid
        return -1
```

## 4. 核心操作

### 4.1 split 操作（拆分区间）

将包含位置 pos 的区间在 pos 处拆分，返回 [pos, ...] 这个新区间。

```python
def split(self, pos):
    """
    在位置 pos 处拆分区间
    返回从 pos 开始的新区间
    """
    if pos > self.n:
        return None

    idx = self._find(pos)
    if idx == -1:
        return None

    node = self.nodes[idx]

    if node.l == pos:
        return node  # 不需要拆分

    # 拆分: [l, pos-1] 和 [pos, r]
    left_node = Node(node.l, pos - 1, node.val)
    right_node = Node(pos, node.r, node.val)

    self.nodes[idx] = left_node
    self.nodes.insert(idx + 1, right_node)

    return right_node
```

### 4.2 assign 操作（区间推平）

将区间 [l, r] 的所有值设为 val。

```python
def assign(self, l, r, val):
    """区间推平：将 [l, r] 全部赋值为 val"""
    # 拆分左右端点
    self.split(l)
    self.split(r + 1)

    # 找到 [l, r] 范围内的所有区间
    left_idx = self._find(l)
    right_idx = self._find(r)

    # 删除 [l, r] 范围内的所有区间
    del self.nodes[left_idx:right_idx]

    # 插入新区间
    self.nodes.insert(left_idx, Node(l, r, val))
```

### 4.3 perform 操作（区间操作）

对 [l, r] 内的每个区间执行操作（如求和、求最大值等）。

```python
def perform(self, l, r, func):
    """对 [l, r] 内的每个区间执行 func(node)"""
    # 确保端点有独立区间
    self.split(l)
    self.split(r + 1)

    left_idx = self._find(l)
    right_idx = self._find(r)

    result = 0
    for i in range(left_idx, right_idx + 1):
        node = self.nodes[i]
        result += func(node)

    return result

def range_sum(self, l, r):
    """区间求和"""
    return self.perform(l, r, lambda node: node.val * (node.r - node.l + 1))

def range_add(self, l, r, val):
    """区间加"""
    self.split(l)
    self.split(r + 1)

    left_idx = self._find(l)
    right_idx = self._find(r)

    for i in range(left_idx, right_idx + 1):
        self.nodes[i].val += val
```

## 5. 完整示例

```python
class ChthollyTreeFull:
    """珂朵莉树完整实现"""

    def __init__(self, data):
        self.n = len(data)
        self.nodes = []
        i = 0
        while i < self.n:
            j = i
            while j < self.n and data[j] == data[i]:
                j += 1
            self.nodes.append(Node(i, j - 1, data[i]))
            i = j

    def _find(self, pos):
        lo, hi = 0, len(self.nodes) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.nodes[mid].r < pos:
                lo = mid + 1
            elif self.nodes[mid].l > pos:
                hi = mid - 1
            else:
                return mid
        return -1

    def split(self, pos):
        if pos > self.n:
            return None
        idx = self._find(pos)
        if idx == -1:
            return None
        node = self.nodes[idx]
        if node.l == pos:
            return node
        self.nodes[idx] = Node(node.l, pos - 1, node.val)
        new_node = Node(pos, node.r, node.val)
        self.nodes.insert(idx + 1, new_node)
        return new_node

    def assign(self, l, r, val):
        self.split(l)
        self.split(r + 1)
        left_idx = self._find(l)
        right_idx = self._find(r)
        del self.nodes[left_idx:right_idx]
        self.nodes.insert(left_idx, Node(l, r, val))

    def range_add(self, l, r, val):
        self.split(l)
        self.split(r + 1)
        left_idx = self._find(l)
        right_idx = self._find(r)
        for i in range(left_idx, right_idx + 1):
            self.nodes[i].val += val

    def range_sum(self, l, r):
        self.split(l)
        self.split(r + 1)
        left_idx = self._find(l)
        right_idx = self._find(r)
        total = 0
        for i in range(left_idx, right_idx + 1):
            node = self.nodes[i]
            total += node.val * (node.r - node.l + 1)
        return total
```

## 6. 使用示例

```python
if __name__ == "__main__":
    data = [1, 1, 2, 2, 3, 3, 3, 4]
    odt = ChthollyTreeFull(data)

    print(f"区间和[1,6] = {odt.range_sum(1, 6)}")  # 1+2+2+3+3+3 = 14

    odt.assign(2, 5, 10)  # 推平 [2,5] 为 10
    print(f"推平后区间和[1,6] = {odt.range_sum(1, 6)}")  # 1+10+10+10+10+3 = 44

    odt.range_add(0, 3, 1)  # [0,3] 每个加1
    print(f"加1后区间和[0,7] = {odt.range_sum(0, 7)}")
```

## 7. 时间复杂度

| 操作 | 最坏 | 数据随机均摊 |
|------|------|-------------|
| split | O(n) | O(log n) |
| assign | O(n) | O(log n) |
| 区间操作 | O(n) | O(log n) |

关键：assign（推平）操作会合并区间，使得区间总数保持在较小范围。在随机数据下，区间数量期望为 O(m / n)，其中 m 是操作次数。

## 8. 适用条件

珂朵莉树高效的前提是**数据随机**，具体而言：
- 区间推平操作足够多
- 操作区间的位置和值是随机的
- 不要构造针对ODT的极端数据

## 9. 与线段树对比

| 特性 | 珂朵莉树 | 线段树 |
|------|---------|--------|
| 实现难度 | 简单 | 中等 |
| 最坏复杂度 | O(n) | O(log n) |
| 随机数据 | 极快 | 稳定 |
| 区间推平 | 天然支持 | 需要懒标记 |
| 适用场景 | 竞赛随机数据 | 通用 |

## 10. 总结

珂朵莉树是一种"投机取巧"但极其高效的数据结构：
- 利用 set 维护连续相同值区间
- 区间推平操作天然减少区间数量
- 在随机数据下表现优异
- 竞赛中的利器，但不适合需要最坏复杂度保证的场景
