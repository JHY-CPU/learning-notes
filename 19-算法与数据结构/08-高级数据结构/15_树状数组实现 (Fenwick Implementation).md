# 树状数组完整实现 (Fenwick Tree Implementation)

## 1. Python 完整实现

### 1.1 基础树状数组

```python
class FenwickTree:
    """树状数组（支持单点修改和前缀和查询）"""

    def __init__(self, n_or_data):
        if isinstance(n_or_data, int):
            self.n = n_or_data
            self.tree = [0] * (n_or_data + 1)
            self.data = [0] * (n_or_data + 1)
        else:
            # 从数组初始化
            self.n = len(n_or_data)
            self.tree = [0] * (self.n + 1)
            self.data = [0] * (self.n + 1)
            for i, val in enumerate(n_or_data):
                self.update(i + 1, val)  # 注意：下标从1开始

    def lowbit(self, x):
        return x & (-x)

    def update(self, idx, val):
        """将 a[idx] 增加 val（下标从1开始）"""
        while idx <= self.n:
            self.tree[idx] += val
            idx += self.lowbit(idx)

    def query(self, idx):
        """查询前缀和 [1, idx]"""
        result = 0
        while idx > 0:
            result += self.tree[idx]
            idx -= self.lowbit(idx)
        return result

    def range_query(self, l, r):
        """查询区间和 [l, r]"""
        return self.query(r) - self.query(l - 1)

    def set_value(self, idx, val):
        """将 a[idx] 设为 val（而非增加）"""
        old = self.range_query(idx, idx)
        self.update(idx, val - old)

    def kth(self, k):
        """查找第k小的元素（需要先作为计数器使用）"""
        idx = 0
        bit_mask = 1 << (self.n.bit_length() - 1)
        while bit_mask:
            next_idx = idx + bit_mask
            if next_idx <= self.n and self.tree[next_idx] < k:
                k -= self.tree[next_idx]
                idx = next_idx
            bit_mask >>= 1
        return idx + 1
```

### 1.2 从数组 O(n) 建树

```python
def build_from_array(self, arr):
    """O(n) 建树"""
    self.n = len(arr)
    self.tree = [0] * (self.n + 1)

    # 先将每个位置设为其管辖的子节点之和
    for i in range(1, self.n + 1):
        self.tree[i] = arr[i - 1]

    # 从前往后，将子节点的值加到父节点
    for i in range(1, self.n + 1):
        j = i + self.lowbit(i)
        if j <= self.n:
            self.tree[j] += self.tree[i]
```

## 2. C++ 完整实现

```cpp
#include <iostream>
#include <cstring>
using namespace std;

const int MAXN = 100005;
int tree[MAXN];
int n;

int lowbit(int x) {
    return x & (-x);
}

void update(int idx, int val) {
    while (idx <= n) {
        tree[idx] += val;
        idx += lowbit(idx);
    }
}

int query(int idx) {
    int result = 0;
    while (idx > 0) {
        result += tree[idx];
        idx -= lowbit(idx);
    }
    return result;
}

int rangeQuery(int l, int r) {
    return query(r) - query(l - 1);
}

// O(n) 建树
void build(int arr[], int size) {
    n = size;
    memset(tree, 0, sizeof(tree));
    for (int i = 1; i <= n; i++) {
        tree[i] = arr[i - 1];
    }
    for (int i = 1; i <= n; i++) {
        int j = i + lowbit(i);
        if (j <= n) {
            tree[j] += tree[i];
        }
    }
}
```

## 3. 逆序对计数（经典应用）

```python
def count_inversions(arr):
    """
    用树状数组统计逆序对数目
    时间复杂度: O(n log n)
    """
    # 离散化
    sorted_vals = sorted(set(arr))
    rank = {v: i + 1 for i, v in enumerate(sorted_vals)}

    n = len(sorted_vals)
    ft = FenwickTree(n)

    inversions = 0
    for i, val in enumerate(arr):
        # 查询已经出现的比当前值大的个数
        r = rank[val]
        inversions += i - ft.query(r)  # 已出现 - 不大于当前的 = 大于当前的
        ft.update(r, 1)  # 标记当前值出现

    return inversions

# 示例
# arr = [5, 2, 6, 1] -> 逆序对: (5,2), (5,1), (2,1), (6,1) = 4个
print(count_inversions([5, 2, 6, 1]))  # 输出: 4
```

## 4. 动态求第K大

```python
def kth_element(ft, k, n):
    """在树状数组中查找第k小的元素"""
    idx = 0
    bit = 1 << (n.bit_length())  # 从最高位开始

    while bit > 0:
        next_idx = idx + bit
        if next_idx <= n and ft.tree[next_idx] < k:
            k -= ft.tree[next_idx]
            idx = next_idx
        bit >>= 1

    return idx + 1
```

## 5. 使用示例

```python
if __name__ == "__main__":
    # 基础操作
    ft = FenwickTree(10)
    ft.update(3, 5)   # a[3] += 5
    ft.update(5, 3)   # a[5] += 3
    ft.update(7, 8)   # a[7] += 8

    print(f"前缀和[1..7] = {ft.query(7)}")    # 5+3+8 = 16
    print(f"前缀和[1..5] = {ft.query(5)}")    # 5+3 = 8
    print(f"区间和[3..7] = {ft.range_query(3, 7)}")  # 5+3+8 = 16
    print(f"区间和[4..6] = {ft.range_query(4, 6)}")  # 3 = 3

    # 从数组初始化
    data = [1, 3, 5, 7, 9]
    ft2 = FenwickTree(data)
    print(f"\n数组 {data} 的前缀和:")
    for i in range(1, 6):
        print(f"  query({i}) = {ft2.query(i)}")

    # 逆序对
    arr = [5, 2, 6, 1]
    print(f"\n{arr} 的逆序对数目: {count_inversions(arr)}")
```

## 6. 二维树状数组

```python
class FenwickTree2D:
    """二维树状数组"""

    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.tree = [[0] * (m + 1) for _ in range(n + 1)]

    def update(self, x, y, val):
        """将 a[x][y] 增加 val"""
        i = x
        while i <= self.n:
            j = y
            while j <= self.m:
                self.tree[i][j] += val
                j += j & (-j)
            i += i & (-i)

    def query(self, x, y):
        """查询前缀和 [1..x][1..y]"""
        result = 0
        i = x
        while i > 0:
            j = y
            while j > 0:
                result += self.tree[i][j]
                j -= j & (-j)
            i -= i & (-i)
        return result

    def range_query(self, x1, y1, x2, y2):
        """查询矩形区域 [x1..x2][y1..y2] 的和"""
        return (self.query(x2, y2) - self.query(x1 - 1, y2)
                - self.query(x2, y1 - 1) + self.query(x1 - 1, y1 - 1))
```

## 7. 复杂度总结

| 操作 | 时间复杂度 |
|------|-----------|
| 单点修改 | O(log n) |
| 前缀和查询 | O(log n) |
| 区间查询 | O(log n) |
| 建树 | O(n) 或 O(n log n) |
| 二维修改 | O(log n * log m) |
| 二维查询 | O(log n * log m) |
