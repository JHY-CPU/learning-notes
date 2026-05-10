# 线段树懒标记 (Segment Tree Lazy Propagation)

## 1. 概述

懒标记（Lazy Propagation）是线段树实现**区间修改**的核心技术。其核心思想：修改操作先"积攒"在节点上，等到需要访问子节点时再下推标记。

## 2. 为什么需要懒标记？

没有懒标记时，区间修改需要更新 O(n) 个节点。懒标记将区间修改优化为 O(log n)。

| 操作 | 无懒标记 | 有懒标记 |
|------|---------|---------|
| 区间修改 | O(n) | O(log n) |
| 区间查询 | O(log n) | O(log n) |

## 3. 懒标记的设计

### 3.1 区间加法

每个节点维护一个 lazy 值，表示"该区间所有元素需要加上的值，但尚未下推到子节点"。

```python
class LazySegmentTree:
    """支持区间加法的懒标记线段树"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)  # 懒标记
        self.data = list(data)
        if self.n > 0:
            self._build(1, 0, self.n - 1)

    def _build(self, node, l, r):
        if l == r:
            self.tree[node] = self.data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid)
        self._build(2 * node + 1, mid + 1, r)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
```

### 3.2 下推操作（Push Down）

```python
    def _push_down(self, node, l, r):
        """将 lazy[node] 下推到子节点"""
        if self.lazy[node] != 0:
            mid = (l + r) // 2

            # 更新左子节点
            self.tree[2 * node] += self.lazy[node] * (mid - l + 1)
            self.lazy[2 * node] += self.lazy[node]

            # 更新右子节点
            self.tree[2 * node + 1] += self.lazy[node] * (r - mid)
            self.lazy[2 * node + 1] += self.lazy[node]

            # 清除当前节点的懒标记
            self.lazy[node] = 0
```

关键点：
- 下推时需要根据子区间长度计算实际的值变化
- 累加到子节点的懒标记上（可能有多个操作叠加）
- 下推后清除当前节点的懒标记

## 4. 区间修改

```python
    def range_update(self, ql, qr, val):
        """区间 [ql, qr] 每个元素加上 val"""
        self._range_update(1, 0, self.n - 1, ql, qr, val)

    def _range_update(self, node, l, r, ql, qr, val):
        if ql <= l and r <= qr:
            # 完全包含：直接更新当前节点
            self.tree[node] += val * (r - l + 1)
            self.lazy[node] += val
            return

        # 部分相交：下推后递归
        self._push_down(node, l, r)
        mid = (l + r) // 2

        if ql <= mid:
            self._range_update(2 * node, l, mid, ql, qr, val)
        if qr > mid:
            self._range_update(2 * node + 1, mid + 1, r, ql, qr, val)

        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
```

## 5. 区间查询（带懒标记）

```python
    def query(self, ql, qr):
        """查询区间 [ql, qr] 的和"""
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.tree[node]

        if r < ql or l > qr:
            return 0

        # 关键：查询前先下推
        self._push_down(node, l, r)
        mid = (l + r) // 2

        return (self._query(2 * node, l, mid, ql, qr) +
                self._query(2 * node + 1, mid + 1, r, ql, qr))
```

## 6. 区间赋值的懒标记

除了区间加法，区间赋值也是一种常见操作：

```python
class AssignLazySegmentTree:
    """支持区间赋值的懒标记线段树"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [None] * (4 * self.n)  # None 表示无赋值操作
        self.data = list(data)
        if self.n > 0:
            self._build(1, 0, self.n - 1)

    def _push_down(self, node, l, r):
        if self.lazy[node] is not None:
            mid = (l + r) // 2
            val = self.lazy[node]

            self.tree[2 * node] = val * (mid - l + 1)
            self.lazy[2 * node] = val

            self.tree[2 * node + 1] = val * (r - mid)
            self.lazy[2 * node + 1] = val

            self.lazy[node] = None

    def range_assign(self, ql, qr, val):
        """区间 [ql, qr] 赋值为 val"""
        self._range_assign(1, 0, self.n - 1, ql, qr, val)

    def _range_assign(self, node, l, r, ql, qr, val):
        if ql <= l and r <= qr:
            self.tree[node] = val * (r - l + 1)
            self.lazy[node] = val
            return

        self._push_down(node, l, r)
        mid = (l + r) // 2

        if ql <= mid:
            self._range_assign(2 * node, l, mid, ql, qr, val)
        if qr > mid:
            self._range_assign(2 * node + 1, mid + 1, r, ql, qr, val)

        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
```

## 7. C++ 懒标记实现

```cpp
const int MAXN = 100005;
long long tree[4 * MAXN];
long long lazy[4 * MAXN];

void pushDown(int node, int l, int r) {
    if (lazy[node] != 0) {
        int mid = (l + r) / 2;
        tree[2 * node] += lazy[node] * (mid - l + 1);
        lazy[2 * node] += lazy[node];
        tree[2 * node + 1] += lazy[node] * (r - mid);
        lazy[2 * node + 1] += lazy[node];
        lazy[node] = 0;
    }
}

void rangeUpdate(int node, int l, int r, int ql, int qr, long long val) {
    if (ql <= l && r <= qr) {
        tree[node] += val * (r - l + 1);
        lazy[node] += val;
        return;
    }
    pushDown(node, l, r);
    int mid = (l + r) / 2;
    if (ql <= mid) rangeUpdate(2 * node, l, mid, ql, qr, val);
    if (qr > mid) rangeUpdate(2 * node + 1, mid + 1, r, ql, qr, val);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

long long query(int node, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return tree[node];
    if (r < ql || l > qr) return 0;
    pushDown(node, l, r);
    int mid = (l + r) / 2;
    return query(2 * node, l, mid, ql, qr)
         + query(2 * node + 1, mid + 1, r, ql, qr);
}
```

## 8. 使用示例

```python
if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    seg = LazySegmentTree(data)

    print(f"原始: query(1,3) = {seg.query(1, 3)}")  # 2+3+4 = 9

    seg.range_update(1, 3, 10)  # [1,3] 每个加10
    print(f"加10后: query(1,3) = {seg.query(1, 3)}")  # 12+13+14 = 39
    print(f"查询: query(0,4) = {seg.query(0, 4)}")    # 1+12+13+14+5 = 45

    seg.range_update(0, 2, 5)   # [0,2] 每个加5
    print(f"再加5: query(0,4) = {seg.query(0, 4)}")   # 6+17+18+14+5 = 60
```

## 9. 懒标记的注意事项

1. **下推时机**：在访问子节点前必须下推
2. **累加处理**：多次区间修改的懒标记要累加
3. **区间长度**：更新节点值时要乘以区间长度
4. **空间开销**：需要额外的 lazy 数组

## 10. 总结

懒标记是线段树支持区间修改的关键：
- 修改时只更新完全包含的节点，记录 lazy 标记
- 查询或进一步修改时，按需下推标记到子节点
- 时间复杂度从 O(n) 优化到 O(log n)
- 支持区间加法、区间赋值等多种操作
