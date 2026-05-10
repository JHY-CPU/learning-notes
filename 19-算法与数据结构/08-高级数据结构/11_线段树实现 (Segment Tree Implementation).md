# 线段树完整实现 (Segment Tree Implementation)

## 1. 完整 Python 实现

### 1.1 区间和线段树

```python
class SegmentTree:
    """区间和线段树（支持单点修改和区间查询）"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.data = list(data)
        if self.n > 0:
            self._build(1, 0, self.n - 1)

    def _build(self, node, l, r):
        """建立线段树"""
        if l == r:
            self.tree[node] = self.data[l]
            return
        mid = (l + r) // 2
        self._build(2 * node, l, mid)
        self._build(2 * node + 1, mid + 1, r)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, val):
        """单点更新：将 data[idx] 改为 val"""
        self.data[idx] = val
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, l, r, idx, val):
        if l == r:
            self.tree[node] = val
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, val)
        self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def query(self, ql, qr):
        """区间查询：返回 data[ql..qr] 的和"""
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.tree[node]
        if r < ql or l > qr:
            return 0
        mid = (l + r) // 2
        return (self._query(2 * node, l, mid, ql, qr) +
                self._query(2 * node + 1, mid + 1, r, ql, qr))
```

### 1.2 区间最大值线段树

```python
class MaxSegmentTree:
    """区间最大值线段树"""

    def __init__(self, data):
        self.n = len(data)
        self.tree = [float('-inf')] * (4 * self.n)
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
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def update(self, idx, val):
        self.data[idx] = val
        self._update(1, 0, self.n - 1, idx, val)

    def _update(self, node, l, r, idx, val):
        if l == r:
            self.tree[node] = val
            return
        mid = (l + r) // 2
        if idx <= mid:
            self._update(2 * node, l, mid, idx, val)
        else:
            self._update(2 * node + 1, mid + 1, r, idx, val)
        self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])

    def query(self, ql, qr):
        return self._query(1, 0, self.n - 1, ql, qr)

    def _query(self, node, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.tree[node]
        if r < ql or l > qr:
            return float('-inf')
        mid = (l + r) // 2
        return max(self._query(2 * node, l, mid, ql, qr),
                   self._query(2 * node + 1, mid + 1, r, ql, qr))
```

## 2. C++ 完整实现

```cpp
#include <iostream>
#include <algorithm>
#include <climits>
using namespace std;

const int MAXN = 100005;
long long tree[4 * MAXN];
int data[MAXN];
int n;

// 建树
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

// 单点更新
void update(int node, int l, int r, int idx, long long val) {
    if (l == r) {
        tree[node] = val;
        return;
    }
    int mid = (l + r) / 2;
    if (idx <= mid)
        update(2 * node, l, mid, idx, val);
    else
        update(2 * node + 1, mid + 1, r, idx, val);
    tree[node] = tree[2 * node] + tree[2 * node + 1];
}

// 区间查询
long long query(int node, int l, int r, int ql, int qr) {
    if (ql <= l && r <= qr) return tree[node];
    if (r < ql || l > qr) return 0;

    int mid = (l + r) / 2;
    return query(2 * node, l, mid, ql, qr)
         + query(2 * node + 1, mid + 1, r, ql, qr);
}

int main() {
    n = 5;
    int arr[] = {1, 3, 5, 7, 9};
    for (int i = 0; i < n; i++) data[i] = arr[i];

    build(1, 0, n - 1);

    cout << "query(1,3) = " << query(1, 0, n-1, 1, 3) << endl;  // 15

    update(1, 0, n-1, 2, 10);
    cout << "after update, query(1,3) = " << query(1, 0, n-1, 1, 3) << endl;  // 20

    return 0;
}
```

## 3. 使用示例

```python
if __name__ == "__main__":
    # 区间和示例
    data = [1, 3, 5, 7, 9, 11]
    seg = SegmentTree(data)

    print(f"原始数组: {data}")
    print(f"query(1,3) = {seg.query(1, 3)}")   # 3+5+7 = 15
    print(f"query(0,5) = {seg.query(0, 5)}")   # 1+3+5+7+9+11 = 36
    print(f"query(2,4) = {seg.query(2, 4)}")   # 5+7+9 = 21

    # 单点修改
    seg.update(2, 10)  # 将 data[2] 改为 10
    print(f"\n修改后 data[2]=10:")
    print(f"query(1,3) = {seg.query(1, 3)}")   # 3+10+7 = 20
    print(f"query(0,5) = {seg.query(0, 5)}")   # 1+3+10+7+9+11 = 41

    # 区间最大值示例
    max_seg = MaxSegmentTree(data)
    print(f"\n区间最大值:")
    print(f"max(0,5) = {max_seg.query(0, 5)}")  # 11
    print(f"max(1,3) = {max_seg.query(1, 3)}")  # 7
```

## 4. 非递归实现（zkw线段树）

```python
class ZkwSegmentTree:
    """zkw线段树（非递归，常数更小）"""

    def __init__(self, data):
        self.n = len(data)
        self.m = 1
        while self.m < self.n:
            self.m <<= 1

        self.tree = [0] * (2 * self.m)

        # 将数据放在叶子层
        for i in range(self.n):
            self.tree[self.m + i] = data[i]

        # 自底向上建树
        for i in range(self.m - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def update(self, idx, val):
        """单点更新"""
        pos = self.m + idx
        self.tree[pos] = val
        pos >>= 1
        while pos > 0:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos >>= 1

    def query(self, l, r):
        """区间查询 [l, r]"""
        res = 0
        l += self.m
        r += self.m

        while l <= r:
            if l & 1:
                res += self.tree[l]
                l += 1
            if not (r & 1):
                res += self.tree[r]
                r -= 1
            l >>= 1
            r >>= 1

        return res
```

## 5. 迭代 vs 递归 性能对比

| 实现方式 | 优点 | 缺点 |
|----------|------|------|
| 递归 | 直观、易理解 | 函数调用开销 |
| 迭代（zkw）| 常数小、速度快 | 实现稍复杂 |

## 6. 总结

线段树的核心操作：
- 建树：O(n)，自底向上合并
- 查询：O(log n)，递归划分区间
- 更新：O(log n)，沿路径向上更新

选择合适的合并函数即可维护不同的区间信息。
