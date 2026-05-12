# 67-数据结构设计-并查集 (Union-Find)

并查集用于处理不相交集合的合并与查询，支持路径压缩和按秩合并优化，近乎 O(1) 操作。

## 完整实现

```javascript
class UnionFind {
  constructor(n) {
    this.parent = Array.from({length: n}, (_, i) => i);
    this.rank = new Array(n).fill(0);
    this.count = n; // 连通分量数量
  }

  // 查找根节点（路径压缩）
  find(x) {
    if (this.parent[x] !== x) {
      this.parent[x] = this.find(this.parent[x]); // 路径压缩
    }
    return this.parent[x];
  }

  // 合并两个集合（按秩合并）
  union(x, y) {
    let px = this.find(x), py = this.find(y);
    if (px === py) return false;
    if (this.rank[px] < this.rank[py]) [px, py] = [py, px];
    this.parent[py] = px;
    if (this.rank[px] === this.rank[py]) this.rank[px]++;
    this.count--;
    return true;
  }

  // 判断是否连通
  connected(x, y) {
    return this.find(x) === this.find(y);
  }

  // 获取连通分量数
  getCount() { return this.count; }
}
```

## C++ 实现

```cpp
#include <vector>
using namespace std;

class UnionFind {
    vector<int> parent, rank;
    int cnt;

public:
    UnionFind(int n) : parent(n), rank(n, 0), cnt(n) {
        for (int i = 0; i < n; i++) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) swap(px, py);
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        cnt--;
        return true;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
    int count() const { return cnt; }
};
```

## 典型应用

```javascript
// 1. 省份数量（判断朋友圈数量）
function findCircleNum(isConnected) {
  const n = isConnected.length;
  const uf = new UnionFind(n);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (isConnected[i][j] === 1) uf.union(i, j);
    }
  }
  return uf.getCount();
}

// 2. 冗余连接（找形成环的边）
function findRedundantConnection(edges) {
  const uf = new UnionFind(edges.length + 1);
  for (const [u, v] of edges) {
    if (!uf.union(u, v)) return [u, v]; // 已连通，是冗余边
  }
}

// 3. 岛屿数量（逐格合并）
function numIslands(grid) {
  const m = grid.length, n = grid[0].length;
  const uf = new UnionFind(m * n);
  let islands = 0;

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      if (grid[i][j] === '1') {
        islands++;
        if (j + 1 < n && grid[i][j+1] === '1') {
          if (uf.union(i*n+j, i*n+j+1)) islands--;
        }
        if (i + 1 < m && grid[i+1][j] === '1') {
          if (uf.union(i*n+j, (i+1)*n+j)) islands--;
        }
      }
    }
  }
  return islands;
}
```

## 时间复杂度

| 操作 | 无优化 | 路径压缩 | 路径压缩+按秩 |
|------|--------|---------|-------------|
| find | O(n) | O(log n) | O(α(n)) ≈ O(1) |
| union | O(n) | O(log n) | O(α(n)) ≈ O(1) |

其中 α(n) 是反阿克曼函数，增长极慢，对所有实际 n 值 α(n) ≤ 5。

## 带权并查集

```javascript
// 维护节点到根的距离（用于等式和不等式判断）
class WeightedUnionFind {
  constructor(n) {
    this.parent = Array.from({length: n}, (_, i) => i);
    this.weight = new Array(n).fill(1.0); // 到父节点的权值
  }

  find(x) {
    if (this.parent[x] !== x) {
      const root = this.find(this.parent[x]);
      this.weight[x] *= this.weight[this.parent[x]];
      this.parent[x] = root;
    }
    return this.parent[x];
  }

  union(x, y, w) {
    // w 表示 x / y 的比值
    const px = this.find(x), py = this.find(y);
    if (px === py) return;
    this.parent[px] = py;
    this.weight[px] = this.weight[y] * w / this.weight[x];
  }

  // 查询 x / y 的比值
  query(x, y) {
    if (this.find(x) !== this.find(y)) return -1.0;
    return this.weight[x] / this.weight[y];
  }
}
```

## 何时使用并查集

- 动态连通性问题
- 判断两个元素是否在同一集合
- 合并两个集合
- 最小生成树（Kruskal 算法）
- 图的连通分量

## 常见陷阱

1. **忘记路径压缩**：不压缩会退化到 O(n)
2. **数组越界**：传入的节点编号超出范围
3. **连通分量计数**：union 成功时要 count--
4. **初始化**：parent[i] = i 不能错
