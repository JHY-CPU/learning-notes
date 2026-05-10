# 左偏树 (Leftist Tree / Leftist Heap)

## 1. 概述

左偏树（Leftist Tree），又称左偏堆，是一种可并堆（Mergeable Heap）。它支持高效的合并操作 O(log n)，同时保持堆的性质。

核心特点：
- 满足堆性质（最小堆或最大堆）
- 支持 O(log n) 合并
- 适合需要频繁合并堆的场景

## 2. dist（距离）的定义

### 2.1 外节点与内节点

- **外节点**：两个子节点中至少有一个为空的节点
- **内节点**：两个子节点都不为空的节点

### 2.2 dist 定义

节点 x 的 dist（也叫 s-value 或 NPL）定义为：x 到其子树中最近外节点的距离。

```
dist(x) = min(dist(left), dist(right)) + 1
dist(NIL) = 0
```

## 3. 左偏性质

左偏树满足以下性质：
- **堆性质**：每个节点的值 <= 其子节点的值（最小堆）
- **左偏性质**：每个节点的左子节点的 dist >= 右子节点的 dist

### 3.1 左偏性质的意义

左偏性质保证右路径是最短路径，合并操作只需沿右路径进行，时间复杂度 O(log n)。

### 3.2 性质推论

- dist(x) = dist(x.right) + 1
- n 个节点的左偏树，dist <= log(n+1)

## 4. 节点定义

### 4.1 Python 实现

```python
class LeftistNode:
    """左偏树节点"""
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.dist = 0  # 到最近外节点的距离
```

### 4.2 C++ 实现

```cpp
struct LeftistNode {
    int val;
    LeftistNode* left;
    LeftistNode* right;
    int dist;

    LeftistNode(int v) : val(v), left(nullptr), right(nullptr), dist(0) {}
};
```

## 5. 合并操作（核心）

### 5.1 合并算法

合并两棵左偏树 A 和 B：
1. 如果 A 为空，返回 B；如果 B 为空，返回 A
2. 确保 A.val <= B.val（最小堆）
3. 递归合并 A.right 和 B
4. 将结果作为 A 的新右子树
5. 维护左偏性质：如果 A.left.dist < A.right.dist，交换左右子树
6. 更新 A.dist

### 5.2 代码实现

```python
def merge(a, b):
    """
    合并两棵左偏树
    返回合并后的根节点
    """
    if a is None:
        return b
    if b is None:
        return a

    # 确保 a.val <= b.val（最小堆）
    if a.val > b.val:
        a, b = b, a

    # 递归合并 a 的右子树和 b
    a.right = merge(a.right, b)

    # 维护左偏性质
    if a.left is None or (a.right is not None and a.left.dist < a.right.dist):
        a.left, a.right = a.right, a.left

    # 更新 dist
    a.dist = (a.right.dist + 1) if a.right else 0

    return a
```

```cpp
LeftistNode* merge(LeftistNode* a, LeftistNode* b) {
    if (!a) return b;
    if (!b) return a;

    if (a->val > b->val) swap(a, b);

    a->right = merge(a->right, b);

    if (!a->left || (a->right && a->left->dist < a->right->dist))
        swap(a->left, a->right);

    a->dist = a->right ? a->right->dist + 1 : 0;

    return a;
}
```

## 6. 插入操作

插入可以看作合并一棵单节点的左偏树：

```python
def insert(self, val):
    """插入新值"""
    new_node = LeftistNode(val)
    self.root = merge(self.root, new_node)
```

## 7. 删除最小值

```python
def pop_min(self):
    """删除并返回最小值"""
    if self.root is None:
        return None

    min_val = self.root.val
    self.root = merge(self.root.left, self.root.right)
    return min_val

def get_min(self):
    """获取最小值"""
    return self.root.val if self.root else None
```

## 8. 完整左偏树类

```python
class LeftistHeap:
    """左偏树（最小堆）"""

    def __init__(self):
        self.root = None

    def is_empty(self):
        return self.root is None

    def insert(self, val):
        new_node = LeftistNode(val)
        self.root = merge(self.root, new_node)

    def get_min(self):
        return self.root.val if self.root else None

    def pop_min(self):
        if not self.root:
            return None
        min_val = self.root.val
        self.root = merge(self.root.left, self.root.right)
        return min_val

    def merge_with(self, other):
        """合并另一棵左偏树"""
        self.root = merge(self.root, other.root)
        other.root = None

    def size(self):
        """计算节点数"""
        return self._size(self.root)

    def _size(self, node):
        if node is None:
            return 0
        return 1 + self._size(node.left) + self._size(node.right)
```

## 9. 使用示例

```python
if __name__ == "__main__":
    # 创建两棵左偏树
    h1 = LeftistHeap()
    for val in [5, 3, 8, 1, 7]:
        h1.insert(val)

    h2 = LeftistHeap()
    for val in [4, 6, 2, 9]:
        h2.insert(val)

    # 合并
    h1.merge_with(h2)

    # 依次弹出最小值
    result = []
    while not h1.is_empty():
        result.append(h1.pop_min())
    print(f"合并后弹出序列: {result}")
    # 输出: [1, 2, 3, 4, 5, 6, 7, 8, 9]
```

## 10. 复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 合并 | O(log n) | 沿右路径合并 |
| 插入 | O(log n) | 合并单节点 |
| 删除最小值 | O(log n) | 合并左右子树 |
| 获取最小值 | O(1) | 查看根节点 |
| 空间 | O(n) | 每节点一个 |

## 11. 左偏树 vs 其他可并堆

| 数据结构 | 合并 | 插入 | 删除最小值 |
|----------|------|------|-----------|
| 左偏树 | O(log n) | O(log n) | O(log n) |
| 斜堆 | O(log n) 均摊 | O(log n) 均摊 | O(log n) 均摊 |
| 二项堆 | O(log n) | O(1) 均摊 | O(log n) |
| 斐波那契堆 | O(1) | O(1) | O(log n) 均摊 |

## 12. 应用场景

1. 可并堆问题：合并多个堆
2. 带删除的优先队列：支持惰性删除
3. 哈夫曼编码：多路归并
4. 竞赛中的特殊堆操作

## 13. 总结

左偏树是一种简洁高效的可并堆：
- 核心操作是合并，时间复杂度 O(log n)
- 左偏性质保证右路径最短，合并只沿右路径进行
- 实现简单，常数小，适合竞赛和工程
