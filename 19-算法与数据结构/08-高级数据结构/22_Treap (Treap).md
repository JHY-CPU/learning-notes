# Treap (Tree + Heap)

## 1. 概述

Treap 是一种结合了**二叉搜索树**和**堆**的随机化数据结构。每个节点同时拥有两个属性：
- **key**：满足BST性质（左 < 根 < 右）
- **priority**：满足堆性质（通常是随机值，最大堆或最小堆）

随机优先级使得Treap的期望高度为 O(log n)。

## 2. 基本原理

### 2.1 双重性质

对于Treap中的每个节点 x：
- x.key 大于左子树中所有节点的 key
- x.key 小于右子树中所有节点的 key（BST性质）
- x.priority 大于其子节点的 priority（最大堆性质）

### 2.2 为什么随机优先级有效

如果 priority 是随机独立均匀分布的，Treap 等价于对 n 个元素进行随机BST构建。随机BST的期望高度为 O(log n)。

## 3. 节点定义

### 3.1 Python 实现

```python
import random

class TreapNode:
    """Treap节点"""
    def __init__(self, key):
        self.key = key
        self.priority = random.random()  # 随机优先级
        self.left = None
        self.right = None
        self.size = 1  # 子树大小（可选，用于按排名操作）

    def __repr__(self):
        return f"TreapNode(key={self.key}, pri={self.priority:.3f})"
```

### 3.2 C++ 实现

```cpp
#include <cstdlib>

struct TreapNode {
    int key;
    double priority;
    TreapNode* left;
    TreapNode* right;
    int size;

    TreapNode(int k) : key(k), priority((double)rand()/RAND_MAX),
                       left(nullptr), right(nullptr), size(1) {}
};
```

## 4. 分裂操作（Split）

### 4.1 按key分裂

将一棵Treap分为两棵：左树所有 key <= val，右树所有 key > val。

```python
def split(root, val):
    """
    按val分裂Treap
    返回 (left, right)，left中所有key<=val
    """
    if root is None:
        return (None, None)

    if root.key <= val:
        left_part, right_part = split(root.right, val)
        root.right = left_part
        update_size(root)
        return (root, right_part)
    else:
        left_part, right_part = split(root.left, val)
        root.left = right_part
        update_size(root)
        return (left_part, root)
```

### 4.2 按排名分裂（Split by rank）

```python
def split_by_rank(root, k):
    """按排名分裂，左树包含前k个元素"""
    if root is None:
        return (None, None)

    left_size = get_size(root.left)

    if k <= left_size:
        left_part, right_part = split_by_rank(root.left, k)
        root.left = right_part
        update_size(root)
        return (left_part, root)
    else:
        left_part, right_part = split_by_rank(root.right, k - left_size - 1)
        root.right = left_part
        update_size(root)
        return (root, right_part)
```

## 5. 合并操作（Merge）

合并两棵Treap，要求左树所有key < 右树所有key。

```python
def merge(left, right):
    """
    合并两棵Treap
    要求：left中所有key < right中所有key
    """
    if left is None:
        return right
    if right is None:
        return left

    if left.priority > right.priority:
        # left作为根
        left.right = merge(left.right, right)
        update_size(left)
        return left
    else:
        # right作为根
        right.left = merge(left, right.left)
        update_size(right)
        return right
```

## 6. 插入操作

```python
def insert(root, key):
    """插入新节点"""
    new_node = TreapNode(key)
    left, right = split(root, key)
    # 合并：left | new_node | right
    return merge(merge(left, new_node), right)
```

## 7. 删除操作

```python
def delete(root, key):
    """删除指定key的节点"""
    # 先分裂出 key 所在的部分
    left, mid_right = split(root, key - 1)
    mid, right = split(mid_right, key)

    # mid 是所有 key==key 的节点（假设不重复则只有一个）
    if mid is not None:
        # 合并 mid 的左右子树
        mid = merge(mid.left, mid.right)

    # 合并回去
    return merge(merge(left, mid), right)
```

## 8. 查找操作

```python
def find(root, key):
    """查找节点"""
    if root is None:
        return None
    if key == root.key:
        return root
    elif key < root.key:
        return find(root.left, key)
    else:
        return find(root.right, key)

def kth(root, k):
    """查找第k小的元素（1-indexed）"""
    if root is None:
        return None

    left_size = get_size(root.left)

    if k <= left_size:
        return kth(root.left, k)
    elif k == left_size + 1:
        return root.key
    else:
        return kth(root.right, k - left_size - 1)

def rank(root, key):
    """查找key的排名（比key小的元素个数+1）"""
    if root is None:
        return 1
    if key <= root.key:
        return rank(root.left, key)
    else:
        return get_size(root.left) + 1 + rank(root.right, key)
```

## 9. 辅助函数

```python
def update_size(node):
    """更新子树大小"""
    if node:
        node.size = 1 + get_size(node.left) + get_size(node.right)

def get_size(node):
    """获取子树大小"""
    return node.size if node else 0
```

## 10. C++ 完整实现

```cpp
TreapNode* merge(TreapNode* left, TreapNode* right) {
    if (!left) return right;
    if (!right) return left;

    if (left->priority > right->priority) {
        left->right = merge(left->right, right);
        left->size = 1 + sz(left->left) + sz(left->right);
        return left;
    } else {
        right->left = merge(left, right->left);
        right->size = 1 + sz(right->left) + sz(right->right);
        return right;
    }
}

pair<TreapNode*, TreapNode*> split(TreapNode* root, int val) {
    if (!root) return {nullptr, nullptr};

    if (root->key <= val) {
        auto [l, r] = split(root->right, val);
        root->right = l;
        root->size = 1 + sz(root->left) + sz(root->right);
        return {root, r};
    } else {
        auto [l, r] = split(root->left, val);
        root->left = r;
        root->size = 1 + sz(root->left) + sz(root->right);
        return {l, root};
    }
}

TreapNode* insert(TreapNode* root, int key) {
    auto [l, r] = split(root, key);
    auto* node = new TreapNode(key);
    return merge(merge(l, node), r);
}
```

## 11. 使用示例

```python
if __name__ == "__main__":
    root = None
    for val in [5, 3, 8, 1, 7, 2, 6]:
        root = insert(root, val)

    print(f"第3小: {kth(root, 3)}")  # 3
    print(f"5的排名: {rank(root, 5)}")  # 5

    root = delete(root, 5)
    print(f"删除5后第5小: {kth(root, 5)}")  # 7
```

## 12. 复杂度分析

| 操作 | 期望时间 | 最坏时间 |
|------|---------|---------|
| 插入 | O(log n) | O(n) |
| 删除 | O(log n) | O(n) |
| 查找 | O(log n) | O(n) |
| 分裂 | O(log n) | O(n) |
| 合并 | O(log n) | O(n) |
| 第k小 | O(log n) | O(n) |

## 13. 应用场景

1. 有序容器：插入、删除、查找
2. 按排名操作：第k小、排名查询
3. 区间翻转：配合split和merge
4. 可持久化数据结构的基础

## 14. 总结

Treap 是一种简洁优雅的随机化平衡树：
- 通过随机优先级维护平衡
- 核心操作是 split 和 merge
- 期望所有操作 O(log n)
- 实现简单，常数小，适合竞赛
