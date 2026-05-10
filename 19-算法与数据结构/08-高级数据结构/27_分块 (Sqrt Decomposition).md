# 分块 (Sqrt Decomposition)

## 1. 概述

分块是一种简单但实用的数据结构技巧，将数组分为若干大小约为 sqrt(n) 的块。每个块维护一些预处理信息，使得区间操作的时间复杂度从 O(n) 降到 O(sqrt(n))。

核心思想：**整块用预处理信息，散块暴力计算**。

## 2. 基本原理

### 2.1 分块策略

将长度为 n 的数组分为 ceil(n / block_size) 个块，每块大小约为 sqrt(n)。

```
数组: [1, 3, 5, 2, 4, 6, 8, 7, 9, 0]
块大小: 3

块0: [1, 3, 5]  -> 块和=9
块1: [2, 4, 6]  -> 块和=12
块2: [8, 7, 9]  -> 块和=24
块3: [0]         -> 块和=0
```

### 2.2 区间操作策略

对于区间 [l, r] 的操作：
- **整块**：直接使用块的预处理信息
- **左散块**：l 所在块的 l 到块尾，逐个处理
- **右散块**：r 所在块的块头到 r，逐个处理

## 3. 完整实现

### 3.1 区间和查询 + 单点修改

```python
import math

class SqrtDecomposition:
    """分块（区间和）"""

    def __init__(self, data):
        self.n = len(data)
        self.data = list(data)
        self.block_size = int(math.sqrt(self.n)) + 1
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.block_sum = [0] * self.num_blocks

        # 预处理块和
        for i in range(self.n):
            self.block_sum[i // self.block_size] += self.data[i]

    def _block_id(self, idx):
        """获取下标所在的块编号"""
        return idx // self.block_size

    def update(self, idx, val):
        """单点修改：data[idx] = val"""
        block = self._block_id(idx)
        self.block_sum[block] += val - self.data[idx]
        self.data[idx] = val

    def query(self, l, r):
        """区间和查询 [l, r]"""
        left_block = self._block_id(l)
        right_block = self._block_id(r)

        if left_block == right_block:
            # 同一块内，暴力求和
            return sum(self.data[l:r + 1])

        result = 0

        # 左散块
        left_end = (left_block + 1) * self.block_size - 1
        for i in range(l, min(left_end + 1, self.n)):
            result += self.data[i]

        # 整块
        for b in range(left_block + 1, right_block):
            result += self.block_sum[b]

        # 右散块
        right_start = right_block * self.block_size
        for i in range(right_start, r + 1):
            result += self.data[i]

        return result
```

### 3.2 区间加法 + 区间查询

```python
class SqrtDecompositionLazy:
    """分块（支持区间加法）"""

    def __init__(self, data):
        self.n = len(data)
        self.data = list(data)
        self.block_size = int(math.sqrt(self.n)) + 1
        self.num_blocks = (self.n + self.block_size - 1) // self.block_size
        self.block_sum = [0] * self.num_blocks
        self.lazy = [0] * self.num_blocks  # 懒标记

        for i in range(self.n):
            self.block_sum[i // self.block_size] += self.data[i]

    def _push_down(self, block):
        """下推懒标记"""
        if self.lazy[block] != 0:
            start = block * self.block_size
            end = min(start + self.block_size, self.n)
            for i in range(start, end):
                self.data[i] += self.lazy[block]
            self.lazy[block] = 0

    def range_add(self, l, r, val):
        """区间 [l, r] 每个元素加 val"""
        left_block = l // self.block_size
        right_block = r // self.block_size

        if left_block == right_block:
            self._push_down(left_block)
            for i in range(l, r + 1):
                self.data[i] += val
                self.block_sum[left_block] += val
            return

        # 左散块
        self._push_down(left_block)
        left_end = (left_block + 1) * self.block_size
        for i in range(l, min(left_end, self.n)):
            self.data[i] += val
            self.block_sum[left_block] += val

        # 整块
        for b in range(left_block + 1, right_block):
            self.lazy[b] += val
            self.block_sum[b] += val * self.block_size

        # 右散块
        self._push_down(right_block)
        right_start = right_block * self.block_size
        for i in range(right_start, r + 1):
            self.data[i] += val
            self.block_sum[right_block] += val

    def query(self, l, r):
        """区间和查询"""
        left_block = l // self.block_size
        right_block = r // self.block_size
        result = 0

        if left_block == right_block:
            self._push_down(left_block)
            return sum(self.data[l:r + 1])

        # 左散块
        self._push_down(left_block)
        left_end = (left_block + 1) * self.block_size
        for i in range(l, min(left_end, self.n)):
            result += self.data[i]

        # 整块
        for b in range(left_block + 1, right_block):
            result += self.block_sum[b]

        # 右散块
        self._push_down(right_block)
        right_start = right_block * self.block_size
        for i in range(right_start, r + 1):
            result += self.data[i]

        return result
```

## 4. C++ 实现

```cpp
const int MAXN = 100005;
int data[MAXN];
long long blockSum[400];  // sqrt(100000) < 400
int blockSize, numBlocks;

void init(int n) {
    blockSize = sqrt(n);
    numBlocks = (n + blockSize - 1) / blockSize;
    for (int i = 0; i < n; i++)
        blockSum[i / blockSize] += data[i];
}

void update(int idx, int val) {
    blockSum[idx / blockSize] += val - data[idx];
    data[idx] = val;
}

long long query(int l, int r) {
    long long result = 0;
    int lb = l / blockSize, rb = r / blockSize;

    if (lb == rb) {
        for (int i = l; i <= r; i++) result += data[i];
        return result;
    }

    for (int i = l; i < (lb + 1) * blockSize; i++) result += data[i];
    for (int b = lb + 1; b < rb; b++) result += blockSum[b];
    for (int i = rb * blockSize; i <= r; i++) result += data[i];

    return result;
}
```

## 5. 使用示例

```python
if __name__ == "__main__":
    data = [1, 3, 5, 2, 4, 6, 8, 7, 9, 0]
    sq = SqrtDecomposition(data)

    print(f"query(0,9) = {sq.query(0, 9)}")  # 45
    print(f"query(2,5) = {sq.query(2, 5)}")  # 5+2+4+6 = 17

    sq.update(3, 10)  # data[3] = 10
    print(f"update后 query(2,5) = {sq.query(2, 5)}")  # 5+10+4+6 = 25

    # 区间加法
    lazy_sq = SqrtDecompositionLazy(data)
    lazy_sq.range_add(1, 4, 3)
    print(f"区间加后 query(0,5) = {lazy_sq.query(0, 5)}")
```

## 6. 块大小的选择

| 块大小 | 整块操作 | 散块操作 | 总复杂度 |
|--------|---------|---------|---------|
| 1 | O(n) | O(0) | O(n) |
| sqrt(n) | O(sqrt(n)) | O(sqrt(n)) | O(sqrt(n)) |
| n | O(1) | O(n) | O(n) |

最优块大小为 sqrt(n)，使整块和散块的开销平衡。

## 7. 与线段树对比

| 特性 | 分块 | 线段树 |
|------|------|--------|
| 实现难度 | 简单 | 中等 |
| 时间复杂度 | O(sqrt(n)) | O(log n) |
| 常数 | 小 | 较大 |
| 空间 | O(n) | O(4n) |
| 灵活性 | 高（容易定制） | 中等 |
| 适用场景 | 通用、竞赛 | 需要log n性能 |

## 8. 莫队算法基础

莫队算法是分块思想在离线查询上的应用，将查询按块排序后分块处理，详见下一节。

## 9. 应用场景

1. 区间和/区间最值查询
2. 区间修改
3. 莫队算法
4. 分块优化DP
5. 替代线段树的简单实现

## 10. 总结

分块是一种"万能"的数据结构技巧：
- 实现简单，容易调试
- 时间复杂度 O(sqrt(n))，略逊于线段树
- 灵活性极高，可以处理各种区间操作
- 是莫队等算法的基础
