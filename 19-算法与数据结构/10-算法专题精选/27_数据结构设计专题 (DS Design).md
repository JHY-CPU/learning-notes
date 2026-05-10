# 数据结构设计专题 (DS Design)

## 一、概念定义与原理

数据结构设计题要求根据需求，组合或改造基本数据结构，实现特定功能。核心在于选择合适的基础结构来满足时间/空间复杂度要求。

---

## 二、经典问题

### 2.1 单调栈结构

维护栈内元素的单调性，$O(1)$ 获取区间最值。

### 2.2 优先队列

支持高效获取最大/最小值。用堆实现 $O(\log n)$ 插入和删除。

### 2.3 线段树

支持区间查询和区间修改，$O(\log n)$ 每次操作。

### 2.4 树状数组 (BIT/Fenwick Tree)

支持单点修改和前缀查询，$O(\log n)$ 每次操作。比线段树代码更简洁。

---

## 三、代码实现

### 3.1 线段树 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

class SegmentTree {
    vector<long long> tree, lazy;
    int n;

    void push_down(int node, int l, int r) {
        if (lazy[node]) {
            int m = (l + r) / 2;
            tree[2*node] += lazy[node] * (m - l + 1);
            tree[2*node+1] += lazy[node] * (r - m);
            lazy[2*node] += lazy[node];
            lazy[2*node+1] += lazy[node];
            lazy[node] = 0;
        }
    }

    void build(vector<int>& a, int node, int l, int r) {
        if (l == r) { tree[node] = a[l]; return; }
        int m = (l + r) / 2;
        build(a, 2*node, l, m);
        build(a, 2*node+1, m+1, r);
        tree[node] = tree[2*node] + tree[2*node+1];
    }

    void update(int node, int l, int r, int ql, int qr, long long val) {
        if (ql <= l && r <= qr) {
            tree[node] += val * (r - l + 1);
            lazy[node] += val;
            return;
        }
        push_down(node, l, r);
        int m = (l + r) / 2;
        if (ql <= m) update(2*node, l, m, ql, qr, val);
        if (qr > m) update(2*node+1, m+1, r, ql, qr, val);
        tree[node] = tree[2*node] + tree[2*node+1];
    }

    long long query(int node, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return tree[node];
        push_down(node, l, r);
        int m = (l + r) / 2;
        long long result = 0;
        if (ql <= m) result += query(2*node, l, m, ql, qr);
        if (qr > m) result += query(2*node+1, m+1, r, ql, qr);
        return result;
    }

public:
    SegmentTree(vector<int>& a) : n(a.size()) {
        tree.resize(4 * n); lazy.resize(4 * n);
        build(a, 1, 0, n - 1);
    }
    void update(int l, int r, long long val) { update(1, 0, n-1, l, r, val); }
    long long query(int l, int r) { return query(1, 0, n-1, l, r); }
};
```

### 3.2 树状数组 - C++

```cpp
class BIT {
    vector<long long> tree;
    int n;
public:
    BIT(int n) : n(n), tree(n + 1, 0) {}

    void update(int i, long long delta) {
        for (; i <= n; i += i & (-i)) tree[i] += delta;
    }

    long long query(int i) {
        long long sum = 0;
        for (; i > 0; i -= i & (-i)) sum += tree[i];
        return sum;
    }

    long long range_query(int l, int r) {
        return query(r) - query(l - 1);
    }
};
```

### 3.3 Python 实现

```python
class BIT:
    """树状数组"""
    def __init__(self, n):
        self.n = n
        self.tree = [0] * (n + 1)

    def update(self, i, delta):
        while i <= self.n:
            self.tree[i] += delta; i += i & (-i)

    def query(self, i):
        s = 0
        while i > 0: s += self.tree[i]; i -= i & (-i)
        return s

    def range_query(self, l, r):
        return self.query(r) - self.query(l - 1)

class SegmentTree:
    """线段树（区间加 + 区间求和）"""
    def __init__(self, data):
        self.n = len(data)
        self.tree = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self._build(data, 1, 0, self.n - 1)

    def _build(self, data, node, l, r):
        if l == r: self.tree[node] = data[l]; return
        m = (l + r) // 2
        self._build(data, 2*node, l, m)
        self._build(data, 2*node+1, m+1, r)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def update(self, l, r, val):
        self._update(1, 0, self.n-1, l, r, val)

    def _update(self, node, l, r, ql, qr, val):
        if ql <= l and r <= qr:
            self.tree[node] += val * (r - l + 1)
            self.lazy[node] += val; return
        m = (l + r) // 2
        if ql <= m: self._update(2*node, l, m, ql, qr, val)
        if qr > m: self._update(2*node+1, m+1, r, ql, qr, val)
        self.tree[node] = self.tree[2*node] + self.tree[2*node+1]

    def query(self, l, r):
        return self._query(1, 0, self.n-1, l, r)

    def _query(self, node, l, r, ql, qr):
        if ql <= l and r <= qr: return self.tree[node]
        m = (l + r) // 2
        result = 0
        if ql <= m: result += self._query(2*node, l, m, ql, qr)
        if qr > m: result += self._query(2*node+1, m+1, r, ql, qr)
        return result

# 测试
bit = BIT(10)
bit.update(1, 3); bit.update(5, 7)
print(bit.range_query(1, 5))  # 10
```

### 3.4 数据流中第K大元素

```cpp
class KthLargest {
    priority_queue<int, vector<int>, greater<int>> pq;
    int k;
public:
    KthLargest(int k, vector<int>& nums) : k(k) {
        for (int x : nums) add(x);
    }
    int add(int val) {
        pq.push(val);
        if (pq.size() > k) pq.pop();
        return pq.top();
    }
};
```

---

## 四、复杂度分析

| 数据结构 | 修改 | 查询 | 空间 |
|---------|------|------|------|
| 树状数组 | $O(\log n)$ | $O(\log n)$ 前缀 | $O(n)$ |
| 线段树 | $O(\log n)$ 区间 | $O(\log n)$ 区间 | $O(4n)$ |
| 优先队列 | $O(\log n)$ | $O(1)$ 最值 | $O(n)$ |
| 平衡BST | $O(\log n)$ | $O(\log n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 703：** 数据流中的第K大元素
2. **LeetCode 307：** 区域和检索 - 数组可修改（线段树/树状数组）
3. **LeetCode 315：** 计算右侧小于当前元素的个数（树状数组）
4. **LeetCode 239：** 滑动窗口最大值（单调队列）
5. **逆序对计数：** 树状数组
