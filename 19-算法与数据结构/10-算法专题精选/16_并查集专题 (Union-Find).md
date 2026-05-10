# 并查集专题 (Union-Find / Disjoint Set Union)

## 一、概念定义与原理

### 1.1 并查集

并查集是一种用于处理**不相交集合**的合并与查询操作的数据结构。支持两种核心操作：

- **Find(x)：** 找到 $x$ 所在集合的代表元素（根节点）
- **Union(x, y)：** 合并 $x$ 和 $y$ 所在的集合

### 1.2 两个优化

1. **路径压缩（Path Compression）：** Find 时将路径上所有节点直接连到根节点
2. **按秩合并（Union by Rank）：** 合并时将较小的树接到较大的树下

两个优化结合后，均摊时间复杂度为 $O(\alpha(n))$，其中 $\alpha$ 是反阿克曼函数，实际视为 $O(1)$。

---

## 二、核心算法

### 2.1 基本实现

每个集合用一棵树表示，根节点为代表元素。用数组 `fa[x]` 记录 $x$ 的父节点，`fa[x] = x` 时 $x$ 为根。

### 2.2 带权并查集

在边上维护额外信息（如与根的差值），支持更复杂的查询。

---

## 三、代码实现

### 3.1 基本并查集 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

class DSU {
    vector<int> fa, rank_;
public:
    DSU(int n) : fa(n), rank_(n, 0) {
        iota(fa.begin(), fa.end(), 0);
    }

    int find(int x) {
        if (fa[x] != x) fa[x] = find(fa[x]); // 路径压缩
        return fa[x];
    }

    bool unite(int x, int y) {
        x = find(x); y = find(y);
        if (x == y) return false;
        if (rank_[x] < rank_[y]) swap(x, y);
        fa[y] = x;
        if (rank_[x] == rank_[y]) rank_[x]++;
        return true;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
};
```

### 3.2 带权并查集 - C++

```cpp
class WeightedDSU {
    vector<int> fa;
    vector<long long> weight; // weight[x]: x 到 fa[x] 的权值
public:
    WeightedDSU(int n) : fa(n), weight(n, 0) {
        iota(fa.begin(), fa.end(), 0);
    }

    pair<int, long long> find(int x) {
        if (fa[x] == x) return {x, 0};
        auto [root, w] = find(fa[x]);
        fa[x] = root;
        weight[x] += w;
        return {fa[x], weight[x]};
    }

    // 告诉你 x 到 y 的权值为 w
    bool unite(int x, int y, long long w) {
        auto [rx, wx] = find(x);
        auto [ry, wy] = find(y);
        if (rx == ry) return wx - wy == w; // 检查一致性
        fa[rx] = ry;
        weight[rx] = wy - wx + w;
        return true;
    }
};
```

### 3.3 Python 实现

```python
class DSU:
    def __init__(self, n):
        self.fa = list(range(n))
        self.rank = [0] * n

    def find(self, x):
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def unite(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return False
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.fa[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# 测试
dsu = DSU(5)
dsu.unite(0, 1)
dsu.unite(2, 3)
print(dsu.connected(0, 1))  # True
print(dsu.connected(0, 2))  # False
dsu.unite(1, 2)
print(dsu.connected(0, 3))  # True
```

### 3.4 连通分量计数

```cpp
int count_components(DSU& dsu, int n) {
    int count = 0;
    for (int i = 0; i < n; i++)
        if (dsu.find(i) == i) count++;
    return count;
}
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| Find（仅路径压缩） | $O(\log n)$ 均摊 | |
| Find（两个优化） | $O(\alpha(n))$ 均摊 | 反阿克曼函数 |
| Union | $O(\alpha(n))$ 均摊 | |
| Connected | $O(\alpha(n))$ 均摊 | |

---

## 五、竞赛与面试应用场景

1. **LeetCode 547：** 省份数量（连通分量）
2. **LeetCode 684：** 冗余连接
3. **LeetCode 721：** 账户合并
4. **Kruskal 最小生成树：** 按边权排序后依次合并
5. **动态连通性：** 离线处理添加边后的连通性查询
