# Splay树 (Splay Tree)

## 1. 概述

Splay树（伸展树）是一种自调整的二叉搜索树，由 Daniel Sleator 和 Robert Tarjan 于 1985 年发明。每次访问节点后，通过**splay操作**将该节点旋转到根。

核心特点：
- 不需要存储平衡信息
- 均摊 O(log n) 的时间复杂度
- 具有"自适应"特性：频繁访问的节点更容易被访问

## 2. Splay操作

### 2.1 基本思想

将目标节点通过一系列旋转逐步移到根。旋转分三种情况：

### 2.2 Zig（单旋转）

当目标节点的父节点是根时：

```
    root              x
   /   \            /   \
  x     C    ->    A     root
 / \                      /   \
A   B                  B      C
```

### 2.3 Zig-Zig（同向双旋转）

目标节点和父节点都是左子节点（或都是右子节点）：

```
        G                x
       / \             /   \
      P   D    ->     A     P
     / \                   /  \
    x   C                 B    G
   / \                       /  \
  A   B                    C    D
```

### 2.4 Zig-Zag（异向双旋转）

目标节点和父节点方向不同：

```
      G                  x
     / \               /   \
    P   D    ->      P      G
   / \              / \    /  \
  A   x            A   B  C    D
     / \
    B   C
```

## 3. Splay实现

### 3.1 Python 实现

```python
class SplayNode:
    """Splay树节点"""
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class SplayTree:
    """Splay树完整实现"""

    def __init__(self):
        self.root = None

    def _rotate_left(self, x):
        """左旋"""
        y = x.right
        x.right = y.left
        if y.left:
            y.left.parent = x
        y.parent = x.parent
        if not x.parent:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def _rotate_right(self, x):
        """右旋"""
        y = x.left
        x.left = y.right
        if y.right:
            y.right.parent = x
        y.parent = x.parent
        if not x.parent:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.right = x
        x.parent = y

    def _splay(self, x):
        """将节点x旋转到根"""
        while x.parent:
            if not x.parent.parent:
                # Zig
                if x == x.parent.left:
                    self._rotate_right(x.parent)
                else:
                    self._rotate_left(x.parent)
            elif x == x.parent.left and x.parent == x.parent.parent.left:
                # Zig-Zig (左左)
                self._rotate_right(x.parent.parent)
                self._rotate_right(x.parent)
            elif x == x.parent.right and x.parent == x.parent.parent.right:
                # Zig-Zig (右右)
                self._rotate_left(x.parent.parent)
                self._rotate_left(x.parent)
            elif x == x.parent.right and x.parent == x.parent.parent.left:
                # Zig-Zag (左右)
                self._rotate_left(x.parent)
                self._rotate_right(x.parent)
            else:
                # Zig-Zag (右左)
                self._rotate_right(x.parent)
                self._rotate_left(x.parent)
```

## 4. 查找操作

```python
def search(self, key):
    """查找key，找到则splay到根"""
    node = self.root
    last = None

    while node:
        last = node
        if key == node.key:
            self._splay(node)
            return node
        elif key < node.key:
            node = node.left
        else:
            node = node.right

    # 未找到，将最近访问的节点splay到根
    if last:
        self._splay(last)
    return None
```

## 5. 插入操作

```python
def insert(self, key):
    """插入新节点"""
    if self.root is None:
        self.root = SplayNode(key)
        return

    node = self.root
    parent = None

    while node:
        parent = node
        if key == node.key:
            self._splay(node)
            return  # 已存在
        elif key < node.key:
            node = node.left
        else:
            node = node.right

    new_node = SplayNode(key)
    new_node.parent = parent

    if key < parent.key:
        parent.left = new_node
    else:
        parent.right = new_node

    self._splay(new_node)
```

## 6. 删除操作

```python
def delete(self, key):
    """删除指定key的节点"""
    node = self.search(key)
    if node is None:
        return

    # node现在在根
    left_tree = node.left
    right_tree = node.right

    if left_tree:
        left_tree.parent = None
    if right_tree:
        right_tree.parent = None

    if left_tree is None:
        self.root = right_tree
    else:
        # 找到左子树的最大值，splay到根
        max_node = left_tree
        while max_node.right:
            max_node = max_node.right
        self._splay(max_node)
        # max_node现在是左子树的根，且没有右子节点
        max_node.right = right_tree
        if right_tree:
            right_tree.parent = max_node
        self.root = max_node
```

## 7. 区间操作（Splay树的优势）

Splay树天然支持高效的区间操作：

```python
def split(self, key):
    """按key分裂：将树分为 <= key 和 > key 两部分"""
    self.search(key)  # 将 <= key 的最大节点splay到根
    if self.root and self.root.key <= key:
        right = self.root.right
        if right:
            right.parent = None
        self.root.right = None
        return (self.root, right)
    else:
        left = self.root.left if self.root else None
        if left:
            left.parent = None
        if self.root:
            self.root.left = None
        return (left, self.root)

def merge_trees(self, left, right):
    """合并两棵Splay树（left所有key < right所有key）"""
    if left is None:
        return right
    if right is None:
        return left

    # 找到left的最大值并splay到根
    max_node = left
    while max_node.right:
        max_node = max_node.right

    # 简单实现：创建辅助函数
    # 实际中需要splay
    root = left
    root.right = right
    right.parent = root
    return root
```

## 8. C++ 实现

```cpp
struct SplayNode {
    int key;
    SplayNode* left;
    SplayNode* right;
    SplayNode* parent;

    SplayNode(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr) {}
};

class SplayTree {
    SplayNode* root;

    void rotateLeft(SplayNode* x) {
        SplayNode* y = x->right;
        x->right = y->left;
        if (y->left) y->left->parent = x;
        y->parent = x->parent;
        if (!x->parent) root = y;
        else if (x == x->parent->left) x->parent->left = y;
        else x->parent->right = y;
        y->left = x;
        x->parent = y;
    }

    void splay(SplayNode* x) {
        while (x->parent) {
            if (!x->parent->parent) {
                if (x == x->parent->left) rotateRight(x->parent);
                else rotateLeft(x->parent);
            } else if (x == x->parent->left && x->parent == x->parent->parent->left) {
                rotateRight(x->parent->parent);
                rotateRight(x->parent);
            } else if (x == x->parent->right && x->parent == x->parent->parent->right) {
                rotateLeft(x->parent->parent);
                rotateLeft(x->parent);
            } else if (x == x->parent->right) {
                rotateLeft(x->parent);
                rotateRight(x->parent);
            } else {
                rotateRight(x->parent);
                rotateLeft(x->parent);
            }
        }
    }
    // ... 其他方法类似
};
```

## 9. 均摊分析

### 9.1 势能函数

```
Phi(T) = sum over all nodes x: rank(x)
rank(x) = log(size(x))
```

### 9.2 均摊复杂度

| 操作 | 最坏 | 均摊 |
|------|------|------|
| 查找 | O(n) | O(log n) |
| 插入 | O(n) | O(log n) |
| 删除 | O(n) | O(log n) |
| Splay | O(n) | O(log n) |

## 10. 与其他平衡树对比

| 特性 | Splay树 | AVL树 | 红黑树 |
|------|---------|-------|--------|
| 最坏单次 | O(n) | O(log n) | O(log n) |
| 均摊复杂度 | O(log n) | O(log n) | O(log n) |
| 额外空间 | 无 | height | color |
| 自适应性 | 有 | 无 | 无 |
| 区间操作 | 天然支持 | 需配合 | 需配合 |

## 11. 应用场景

1. 自适应访问模式
2. 区间翻转（文艺树）
3. 可合并有序集合
4. 网络路由器的缓存管理
5. 编译器的符号表

## 12. 总结

Splay树通过将访问的节点伸展到根，实现了优秀的均摊性能。它不需要额外存储平衡信息，且天然支持区间操作。缺点是单次操作可能退化到 O(n)，不适合实时性要求高的场景。
