# KD树 (KD-Tree)

## 1. 概述

KD树（K-Dimensional Tree）是一种用于组织K维空间中点的数据结构。它是二叉搜索树在多维空间的推广，广泛应用于多维搜索、最近邻查询、范围查询等场景。

## 2. 基本原理

### 2.1 空间划分

KD树通过交替使用各维度对空间进行划分：
- 第1层：按第1维（x）划分
- 第2层：按第2维（y）划分
- ...
- 第k层：按第k维划分
- 第k+1层：按第1维划分（循环）

### 2.2 二维示例

```
点集: [(2,3), (5,4), (9,6), (4,7), (8,1), (7,2)]

KD树:              空间划分:
      (7,2)          y
      /   \          ^
     /     \         |
  (5,4)   (9,6)     +-----+------+
   / \               |     |      |
(2,3)(4,7)(8,1)     |  A  |  B   |
                     +-----+------+
                     0     7      x
```

## 3. 节点定义

### 3.1 Python 实现

```python
class KDNode:
    """KD树节点"""
    def __init__(self, point, split_dim=0):
        self.point = point       # K维坐标
        self.split_dim = split_dim  # 分割维度
        self.left = None
        self.right = None

    def __repr__(self):
        return f"KDNode({self.point})"
```

### 3.2 C++ 实现

```cpp
const int K = 2;  // 维度

struct KDNode {
    int point[K];          // K维坐标
    int split_dim;         // 分割维度
    KDNode* left;
    KDNode* right;

    KDNode() : left(nullptr), right(nullptr), split_dim(0) {}
};
```

## 4. 建树操作

### 4.1 策略

选择方差最大的维度作为分割维度，取中位数作为分割点。

```python
def build_kd_tree(points, depth=0):
    """
    构建KD树
    points: K维点列表
    depth: 当前深度
    """
    if not points:
        return None

    k = len(points[0])  # 维度
    split_dim = depth % k

    # 按当前维度排序
    points.sort(key=lambda p: p[split_dim])

    # 取中位数作为分割点
    mid = len(points) // 2
    node = KDNode(points[mid], split_dim)

    # 递归构建左右子树
    node.left = build_kd_tree(points[:mid], depth + 1)
    node.right = build_kd_tree(points[mid + 1:], depth + 1)

    return node
```

### 4.2 优化版本（选择方差最大的维度）

```python
import statistics

def build_kd_tree_optimized(points, depth=0):
    """优化版本：选择方差最大的维度"""
    if not points:
        return None

    k = len(points[0])

    # 选择方差最大的维度
    if len(points) > 1:
        variances = []
        for dim in range(k):
            vals = [p[dim] for p in points]
            variances.append(statistics.variance(vals))
        split_dim = variances.index(max(variances))
    else:
        split_dim = depth % k

    points.sort(key=lambda p: p[split_dim])
    mid = len(points) // 2

    node = KDNode(points[mid], split_dim)
    node.left = build_kd_tree_optimized(points[:mid], depth + 1)
    node.right = build_kd_tree_optimized(points[mid + 1:], depth + 1)

    return node
```

## 5. 最近邻搜索

### 5.1 算法

1. 从根节点开始，沿分割维度递归到叶节点
2. 回溯时检查另一侧子树是否有更近的点
3. 如果当前点到分割面的距离 >= 已知最近距离，则剪枝

```python
import math

def nearest_neighbor(root, target, best=None):
    """
    最近邻搜索
    返回距离target最近的点
    """
    if root is None:
        return best

    # 更新最近点
    if best is None or distance(root.point, target) < distance(best, target):
        best = root.point

    dim = root.split_dim
    diff = target[dim] - root.point[dim]

    # 优先搜索目标所在的那一侧
    if diff <= 0:
        good_side = root.left
        bad_side = root.right
    else:
        good_side = root.right
        bad_side = root.left

    best = nearest_neighbor(good_side, target, best)

    # 检查另一侧是否有更近的点
    if abs(diff) < distance(best, target):
        best = nearest_neighbor(bad_side, target, best)

    return best

def distance(p1, p2):
    """欧几里得距离的平方"""
    return sum((a - b) ** 2 for a, b in zip(p1, p2))
```

### 5.2 C++ 实现

```cpp
double dist(int a[], int b[]) {
    double d = 0;
    for (int i = 0; i < K; i++)
        d += (a[i] - b[i]) * (a[i] - b[i]);
    return d;
}

void nearestNeighbor(KDNode* root, int target[], int best[], double& bestDist) {
    if (!root) return;

    double d = dist(root->point, target);
    if (d < bestDist) {
        bestDist = d;
        memcpy(best, root->point, sizeof(int) * K);
    }

    int dim = root->split_dim;
    int diff = target[dim] - root->point[dim];

    KDNode* good = (diff <= 0) ? root->left : root->right;
    KDNode* bad = (diff <= 0) ? root->right : root->left;

    nearestNeighbor(good, target, best, bestDist);

    if ((double)(diff * diff) < bestDist)
        nearestNeighbor(bad, target, best, bestDist);
}
```

## 6. 范围查询

```python
def range_query(root, low, high):
    """
    范围查询：返回所有在 [low, high] 范围内的点
    low, high: K维的下界和上界
    """
    if root is None:
        return []

    result = []

    # 检查当前点是否在范围内
    if all(low[i] <= root.point[i] <= high[i] for i in range(len(root.point))):
        result.append(root.point)

    dim = root.split_dim

    # 如果当前维度的值 >= low[dim]，左子树可能有满足条件的点
    if root.point[dim] >= low[dim]:
        result.extend(range_query(root.left, low, high))

    # 如果当前维度的值 <= high[dim]，右子树可能有满足条件的点
    if root.point[dim] <= high[dim]:
        result.extend(range_query(root.right, low, high))

    return result
```

## 7. 插入操作

```python
def insert(root, point, depth=0):
    """插入新点"""
    if root is None:
        return KDNode(point, depth % len(point))

    dim = depth % len(point)

    if point[dim] < root.point[dim]:
        root.left = insert(root.left, point, depth + 1)
    else:
        root.right = insert(root.right, point, depth + 1)

    return root
```

## 8. 完整使用示例

```python
if __name__ == "__main__":
    points = [(2, 3), (5, 4), (9, 6), (4, 7), (8, 1), (7, 2)]

    # 构建KD树
    root = build_kd_tree(points)

    # 最近邻搜索
    target = (6, 3)
    nearest = nearest_neighbor(root, target)
    print(f"距离 {target} 最近的点: {nearest}")

    # 范围查询
    low = (3, 2)
    high = (8, 5)
    in_range = range_query(root, low, high)
    print(f"范围 {low}~{high} 内的点: {in_range}")

    # 插入
    root = insert(root, (6, 5))
    print(f"插入 (6,5) 后最近邻: {nearest_neighbor(root, target)}")
```

## 9. 时间复杂度

| 操作 | 平均 | 最坏 |
|------|------|------|
| 建树 | O(n log n) | O(n log n) |
| 最近邻 | O(log n) | O(n) |
| 范围查询 | O(n^(1-1/k) + m) | O(n) |
| 插入 | O(log n) | O(n) |

其中 k 是维度，m 是结果数量。

## 10. 与其他结构对比

| 数据结构 | 最近邻 | 范围查询 | 适用维度 |
|----------|--------|---------|---------|
| KD树 | O(log n) 平均 | O(n^(1-1/k)) | 低维（k<=20） |
| R树 | O(log n) | O(log n) | 二维/三维 |
| 暴力 | O(n) | O(n) | 任意 |

## 11. 应用场景

1. 最近邻搜索（KNN分类）
2. 点云数据处理
3. 计算几何
4. 图像检索
5. 碰撞检测

## 12. 总结

KD树是多维空间搜索的基本数据结构：
- 通过交替维度划分空间
- 支持最近邻、范围查询等操作
- 在低维空间中效率很高
- 高维时退化严重（维度灾难）
