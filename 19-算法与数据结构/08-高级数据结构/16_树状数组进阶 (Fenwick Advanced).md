# 树状数组进阶 (Fenwick Tree Advanced)

## 1. 概述

树状数组虽然结构简单，但通过一些技巧可以扩展到更复杂的场景：区间修改+单点查询、区间修改+区间查询、二维BIT等。

## 2. 区间修改 + 单点查询

### 2.1 差分思想

用差分数组 d[i] = a[i] - a[i-1] 来维护。区间 [l, r] 加 val 等价于：
- d[l] += val
- d[r+1] -= val

单点查询 a[i] = d[1] + d[2] + ... + d[i]，即差分数组的前缀和。

### 2.2 实现

```python
class BITRangeUpdate:
    """区间修改 + 单点查询的树状数组"""

    def __init__(self, n):
        self.n = n
        self.bit = FenwickTree(n)

    def range_add(self, l, r, val):
        """区间 [l, r] 每个元素加 val"""
        self.bit.update(l, val)      # d[l] += val
        if r + 1 <= self.n:
            self.bit.update(r + 1, -val)  # d[r+1] -= val

    def point_query(self, idx):
        """查询 a[idx] 的值"""
        return self.bit.query(idx)  # 差分的前缀和

    def build_from_array(self, arr):
        """从数组初始化"""
        for i in range(len(arr)):
            self.range_add(i + 1, i + 1, arr[i])
```

## 3. 区间修改 + 区间查询

### 3.1 数学推导

我们需要维护两个树状数组 B1 和 B2，使得：

```
sum[1..x] = x * query(B1, x) - query(B2, x)
```

区间 [l, r] 加 val 时：
- B1: update(l, val), update(r+1, -val)
- B2: update(l, val*(l-1)), update(r+1, -val*r)

### 3.2 完整实现

```python
class BITRangeUpdateRangeQuery:
    """区间修改 + 区间查询的树状数组"""

    def __init__(self, n):
        self.n = n
        self.b1 = FenwickTree(n)  # 维护差分
        self.b2 = FenwickTree(n)  # 维护 i*d[i]

    def _update(self, idx, val):
        """内部更新"""
        self.b1.update(idx, val)
        self.b2.update(idx, val * (idx - 1))

    def range_add(self, l, r, val):
        """区间 [l, r] 每个元素加 val"""
        self._update(l, val)
        if r + 1 <= self.n:
            self._update(r + 1, -val)

    def prefix_sum(self, idx):
        """查询前缀和 [1, idx]"""
        return idx * self.b1.query(idx) - self.b2.query(idx)

    def range_sum(self, l, r):
        """查询区间和 [l, r]"""
        return self.prefix_sum(r) - self.prefix_sum(l - 1)

    def build_from_array(self, arr):
        """从数组初始化"""
        for i, val in enumerate(arr):
            self.range_add(i + 1, i + 1, val)
```

### 3.3 C++ 实现

```cpp
class BIT_RURQ {
    int n;
    FenwickTree b1, b2;

public:
    BIT_RURQ(int size) : n(size), b1(size), b2(size) {}

    void _update(int idx, long long val) {
        b1.update(idx, val);
        b2.update(idx, val * (idx - 1));
    }

    void rangeAdd(int l, int r, long long val) {
        _update(l, val);
        if (r + 1 <= n) _update(r + 1, -val);
    }

    long long prefixSum(int idx) {
        return idx * b1.query(idx) - b2.query(idx);
    }

    long long rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```

## 4. 二维树状数组（区间修改+区间查询）

### 4.1 原理

需要维护 4 个树状数组，扩展一维的推导到二维。

### 4.2 实现

```python
class BIT2DRangeUpdate:
    """二维区间修改 + 区间查询"""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        # 维护 4 个二维BIT
        self.t1 = [[0] * (m + 1) for _ in range(n + 1)]
        self.t2 = [[0] * (m + 1) for _ in range(n + 1)]
        self.t3 = [[0] * (m + 1) for _ in range(n + 1)]
        self.t4 = [[0] * (m + 1) for _ in range(n + 1)]

    def _update(self, t, x, y, val):
        i = x
        while i <= self.n:
            j = y
            while j <= self.m:
                t[i][j] += val
                j += j & (-j)
            i += i & (-i)

    def _query(self, t, x, y):
        res = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                res += t[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return res

    def range_add(self, x1, y1, x2, y2, val):
        """矩形 [x1,y1]-[x2,y2] 每个元素加 val"""
        self._update(self.t1, x1, y1, val)
        self._update(self.t1, x1, y2 + 1, -val)
        self._update(self.t1, x2 + 1, y1, -val)
        self._update(self.t1, x2 + 1, y2 + 1, val)

        self._update(self.t2, x1, y1, val * (x1 - 1))
        self._update(self.t2, x1, y2 + 1, -val * (x1 - 1))
        self._update(self.t2, x2 + 1, y1, -val * x2)
        self._update(self.t2, x2 + 1, y2 + 1, val * x2)

        self._update(self.t3, x1, y1, val * (y1 - 1))
        self._update(self.t3, x1, y2 + 1, -val * y2)
        self._update(self.t3, x2 + 1, y1, -val * (y1 - 1))
        self._update(self.t3, x2 + 1, y2 + 1, val * y2)

        self._update(self.t4, x1, y1, val * (x1 - 1) * (y1 - 1))
        self._update(self.t4, x1, y2 + 1, -val * (x1 - 1) * y2)
        self._update(self.t4, x2 + 1, y1, -val * x2 * (y1 - 1))
        self._update(self.t4, x2 + 1, y2 + 1, val * x2 * y2)
```

## 5. 树状数组上二分

在权值树状数组上查找第 k 小：

```python
def find_kth(ft, k):
    """查找第 k 小的元素值"""
    idx = 0
    bit = 1
    # 找到 >= n 的最大2的幂
    while bit <= ft.n:
        bit <<= 1
    bit >>= 1

    while bit > 0:
        next_idx = idx + bit
        if next_idx <= ft.n and ft.tree[next_idx] < k:
            k -= ft.tree[next_idx]
            idx = next_idx
        bit >>= 1

    return idx + 1
```

## 6. 使用示例

```python
if __name__ == "__main__":
    # 区间修改 + 区间查询
    bit = BITRangeUpdateRangeQuery(10)
    bit.range_add(1, 5, 3)   # [1,5] 每个加3
    bit.range_add(3, 8, 2)   # [3,8] 每个加2

    print(f"区间和[1..5] = {bit.range_sum(1, 5)}")   # 3+3+5+5+5 = 21
    print(f"区间和[3..7] = {bit.range_sum(3, 7)}")   # 5+5+5+2+2 = 19
    print(f"前缀和[1..8] = {bit.prefix_sum(8)}")     # 完整计算
```

## 7. 应用场景

| 功能 | 方法 |
|------|------|
| 单点修改+前缀和 | 基础BIT |
| 区间修改+单点查询 | 差分BIT |
| 区间修改+区间查询 | 双BIT |
| 二维区间修改+查询 | 四BIT |
| 动态第K小 | 权值BIT+二分 |

## 8. 总结

通过差分和数学技巧，树状数组可以支持更复杂的区间操作：
- 区间修改+单点查询：使用差分思想
- 区间修改+区间查询：维护两个BIT
- 二维扩展：维护4个BIT
- 代码量仍然远少于线段树
