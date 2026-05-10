# 线段树基础 (Segment Tree Basics)

## 1. 概述

线段树是一种用于维护**区间信息**的树形数据结构。它支持高效的区间查询和区间修改操作，广泛应用于算法竞赛和实际工程中。

线段树的核心思想：将一个大区间递归地划分为若干小区间，每个节点维护对应区间的信息（如区间和、区间最大值等）。

## 2. 基本概念

### 2.1 区间表示

对于长度为 n 的数组 a[0..n-1]，线段树中的每个节点对应一个区间 [l, r]。

### 2.2 树的结构

```
区间 [0,7] 的线段树：

           [0,7] (根)
          /      \
      [0,3]      [4,7]
      /    \     /    \
   [0,1] [2,3] [4,5] [6,7]
   /  \   /  \  /  \  /  \
  [0] [1][2][3][4][5][6][7]  (叶子)
```

### 2.3 节点编号规则

使用数组模拟时，对于节点 u：
- 左子节点：2*u
- 右子节点：2*u + 1
- 父节点：u / 2（整数除法）

## 3. 建树操作

### 3.1 原理

将数组建为线段树，每个叶子节点存储单个元素，内部节点存储子节点信息的合并结果。

### 3.2 代码实现（区间和）

```python
class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)  # 4倍空间足够
        self.data = data
        self._build(1, 0, self.n - 1)

    def _build(self, node, l, r):
        """建立线段树
        node: 当前节点编号
        l, r: 当前节点对应的区间
        """
        if l == r:
            # 叶子节点
            self.tree[node] = self.data[l]
            return

        mid = (l + r) // 2
        self._build(2 * node, l, mid)        # 建左子树
        self._build(2 * node + 1, mid + 1, r) # 建右子树

        # 合并子节点信息
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
```

### 3.3 C++ 实现

```cpp
const int MAXN = 100005;
long long tree[4 * MAXN];
int data[MAXN];

void build(int node, int l, int r) {
    if (l == r) {
        tree[node] = data[l];
        return;
    }
    int mid = (l + r) / 2;
    build(2 * node, l, mid);
    build(2 * node + 1, mid + 1, r);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}
```

## 4. 区间查询

### 4.1 原理

查询区间 [ql, qr] 的信息时，递归地将查询区间与当前节点区间比较：
- 完全包含：直接返回当前节点的值
- 不相交：返回中性元素（如区间和返回0）
- 部分相交：递归查询左右子节点，合并结果

### 4.2 代码实现

```python
def query(self, ql, qr):
    """查询区间 [ql, qr] 的和"""
    return self._query(1, 0, self.n - 1, ql, qr)

def _query(self, node, l, r, ql, qr):
    """
    node: 当前节点
    l, r: 当前节点区间
    ql, qr: 查询区间
    """
    if ql <= l and r <= qr:
        # 完全包含
        return self.tree[node]

    if r < ql or l > qr:
        # 不相交
        return 0

    # 部分相交
    mid = (l + r) // 2
    left_sum = self._query(2 * node, l, mid, ql, qr)
    right_sum = self._query(2 * node + 1, mid + 1, r, ql, qr)
    return left_sum + right_sum
```

### 4.3 C++ 实现

```cpp
long long query(int node, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return tree[node];
    if (r < ql || l > qr) return 0;

    int mid = (l + r) / 2;
    return query(2 * node, l, mid, ql, qr)
         + query(2 * node + 1, mid + 1, r, ql, qr);
}
```

## 5. 单点更新

### 5.1 原理

将 a[idx] 修改为 new_val，然后沿着路径向上更新所有祖先节点的值。

### 5.2 代码实现

```python
def update(self, idx, val):
    """将 a[idx] 更新为 val"""
    self.data[idx] = val
    self._update(1, 0, self.n - 1, idx, val)

def _update(self, node, l, r, idx, val):
    if l == r:
        # 叶子节点
        self.tree[node] = val
        return

    mid = (l + r) // 2
    if idx <= mid:
        self._update(2 * node, l, mid, idx, val)
    else:
        self._update(2 * node + 1, mid + 1, r, idx, val)

    # 更新当前节点
    self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
```

## 6. 空间分析

线段树需要 4*n 的空间：
- 完全二叉树：2*n - 1 个节点
- 4*n 是安全上界（含偏移和不对齐）

| 数组长度 n | 节点数 | 空间 (long long) |
|-----------|--------|-----------------|
| 10^5 | ~4*10^5 | ~3.2 MB |
| 10^6 | ~4*10^6 | ~32 MB |
| 10^7 | ~4*10^7 | ~320 MB |

## 7. 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 建树 | O(n) |
| 区间查询 | O(log n) |
| 单点更新 | O(log n) |
| 区间修改（懒标记）| O(log n) |

## 8. 不同维护信息的线段树

线段树可以维护不同的区间信息，只需修改合并操作：

| 维护信息 | 合并操作 | 中性元素 |
|----------|---------|---------|
| 区间和 | a + b | 0 |
| 区间最大值 | max(a, b) | -inf |
| 区间最小值 | min(a, b) | +inf |
| 区间GCD | gcd(a, b) | 0 |
| 区间乘积 | a * b | 1 |

## 9. 应用场景

1. 区间和查询 + 单点修改
2. 区间最值查询
3. 逆序对计数（离散化 + 线段树）
4. 区间修改 + 区间查询（懒标记）
5. 动态第K大（权值线段树）

## 10. 总结

线段树是一种强大的区间数据结构：
- 建树 O(n)，查询和修改 O(log n)
- 通过递归划分区间实现高效的区间操作
- 可以维护各种区间信息（和、最值、GCD等）
- 配合懒标记可以实现区间修改
