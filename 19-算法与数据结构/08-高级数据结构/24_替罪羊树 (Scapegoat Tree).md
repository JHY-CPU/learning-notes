# 替罪羊树 (Scapegoat Tree)

## 1. 概述

替罪羊树（Scapegoat Tree）由 Igal Galperin 和 Ronald L. Rivest 于 1993 年提出。它不使用旋转操作，而是在检测到不平衡时，将整个子树**重建**为完全二叉树。

核心思想：当某个节点的子树大小与其子节点的子树大小之比超过阈值时，将该子树重建。

## 2. 平衡参数 alpha

### 2.1 定义

参数 alpha (0.5 < alpha < 1) 控制平衡的严格程度：
- alpha = 0.5：要求完全平衡（像完全二叉树）
- alpha 接近 1：几乎不重建

### 2.2 平衡条件

对于任意节点 x，需要满足：
- size(x.left) <= alpha * size(x)
- size(x.right) <= alpha * size(x)

即：**每个子节点的子树大小不超过当前节点子树大小的 alpha 倍**。

### 2.3 常用取值

alpha = 0.75 或 alpha = 2/3 是常见的选择，平衡了重建频率和树的高度。

## 3. 重建操作

### 3.1 原理

重建是将一棵子树展开为有序数组，再构建为完全二叉树。

```python
def rebuild(self, node):
    """重建以node为根的子树"""
    # 1. 中序遍历收集所有节点
    nodes = []
    self._inorder_collect(node, nodes)

    # 2. 从有序数组重建为平衡BST
    new_root = self._build_balanced(nodes, 0, len(nodes) - 1)

    # 3. 更新父节点指针
    if node.parent is None:
        self.root = new_root
    elif node == node.parent.left:
        node.parent.left = new_root
    else:
        node.parent.right = new_root

    if new_root:
        new_root.parent = node.parent

def _inorder_collect(self, node, result):
    """中序遍历收集节点"""
    if node is None:
        return
    self._inorder_collect(node.left, result)
    result.append(node)
    self._inorder_collect(node.right, result)

def _build_balanced(self, nodes, l, r):
    """从有序节点列表构建平衡BST"""
    if l > r:
        return None

    mid = (l + r) // 2
    node = nodes[mid]

    node.left = self._build_balanced(nodes, l, mid - 1)
    node.right = self._build_balanced(nodes, mid + 1, r)

    if node.left:
        node.left.parent = node
    if node.right:
        node.right.parent = node

    return node
```

## 4. 完整实现

### 4.1 Python 实现

```python
class ScapegoatNode:
    """替罪羊树节点"""
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class ScapegoatTree:
    """替罪羊树"""

    def __init__(self, alpha=0.75):
        self.root = None
        self.alpha = alpha
        self.size = 0  # 总节点数
        self.max_size = 0  # 历史最大节点数（用于删除后重建）

    def _size(self, node):
        """计算子树大小"""
        if node is None:
            return 0
        return 1 + self._size(node.left) + self._size(node.right)

    def _is_balanced(self, node):
        """检查节点是否平衡"""
        if node is None:
            return True
        left_size = self._size(node.left)
        right_size = self._size(node.right)
        total = 1 + left_size + right_size
        return (left_size <= self.alpha * total and
                right_size <= self.alpha * total)

    def _find_scapegoat(self, node):
        """沿路径向上找替罪羊节点"""
        while node is not None:
            if not self._is_balanced(node):
                return node
            node = node.parent
        return None

    def insert(self, key):
        """插入新节点"""
        new_node = ScapegoatNode(key)

        if self.root is None:
            self.root = new_node
            self.size = 1
            self.max_size = 1
            return

        # 标准BST插入
        parent = None
        current = self.root
        path = []  # 记录插入路径

        while current is not None:
            parent = current
            path.append(current)
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                return  # 重复

        new_node.parent = parent
        if key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node

        path.append(new_node)
        self.size += 1
        self.max_size = max(self.max_size, self.size)

        # 检查是否需要重建
        for node in reversed(path):
            if not self._is_balanced(node):
                scapegoat = self._find_scapegoat(node)
                if scapegoat:
                    self.rebuild(scapegoat)
                break

    def delete(self, key):
        """删除节点"""
        node = self._search(key)
        if node is None:
            return

        # BST删除
        if node.left is None or node.right is None:
            self._replace(node, node.left or node.right)
        else:
            successor = self._find_min(node.right)
            node.key = successor.key
            self._replace(successor, successor.right)

        self.size -= 1

        # 如果size过小，重建整棵树
        if self.size < self.alpha * self.max_size:
            if self.root:
                self.rebuild(self.root)
            self.max_size = self.size

    def _search(self, key):
        """查找节点"""
        node = self.root
        while node:
            if key == node.key:
                return node
            elif key < node.key:
                node = node.left
            else:
                node = node.right
        return None

    def _find_min(self, node):
        """找最小节点"""
        while node.left:
            node = node.left
        return node

    def _replace(self, old, new):
        """用new替换old"""
        if old.parent is None:
            self.root = new
        elif old == old.parent.left:
            old.parent.left = new
        else:
            old.parent.right = new
        if new:
            new.parent = old.parent

    # rebuild 和 _build_balanced 已在上方定义
```

### 4.2 C++ 实现

```cpp
struct ScapegoatNode {
    int key;
    ScapegoatNode* left;
    ScapegoatNode* right;
    ScapegoatNode* parent;

    ScapegoatNode(int k) : key(k), left(nullptr), right(nullptr), parent(nullptr) {}
};

class ScapegoatTree {
    ScapegoatNode* root;
    double alpha;
    int size;

    int getSize(ScapegoatNode* node) {
        if (!node) return 0;
        return 1 + getSize(node->left) + getSize(node->right);
    }

    bool isBalanced(ScapegoatNode* node) {
        if (!node) return true;
        int l = getSize(node->left);
        int r = getSize(node->right);
        int total = 1 + l + r;
        return l <= alpha * total && r <= alpha * total;
    }

    void inorderCollect(ScapegoatNode* node, vector<ScapegoatNode*>& nodes) {
        if (!node) return;
        inorderCollect(node->left, nodes);
        nodes.push_back(node);
        inorderCollect(node->right, nodes);
    }

    ScapegoatNode* buildBalanced(vector<ScapegoatNode*>& nodes, int l, int r) {
        if (l > r) return nullptr;
        int mid = (l + r) / 2;
        ScapegoatNode* node = nodes[mid];
        node->left = buildBalanced(nodes, l, mid - 1);
        node->right = buildBalanced(nodes, mid + 1, r);
        if (node->left) node->left->parent = node;
        if (node->right) node->right->parent = node;
        return node;
    }

    void rebuild(ScapegoatNode* node) {
        vector<ScapegoatNode*> nodes;
        inorderCollect(node, nodes);
        ScapegoatNode* newRoot = buildBalanced(nodes, 0, nodes.size() - 1);
        // ... 更新父指针
    }
};
```

## 5. 复杂度分析

| 操作 | 最坏（单次） | 均摊 |
|------|------------|------|
| 查找 | O(log n) | O(log n) |
| 插入 | O(n)（重建时） | O(log n) |
| 删除 | O(n)（重建时） | O(log n) |
| 重建 | O(k)，k为子树大小 | - |

## 6. alpha 的影响

| alpha值 | 树高上界 | 重建频率 | 适用场景 |
|---------|---------|---------|---------|
| 0.5 | log n / log 2 ≈ log n | 非常频繁 | 不实用 |
| 2/3 | log n / log(3/2) ≈ 1.7 log n | 适中 | 常用 |
| 0.75 | log n / log(4/3) ≈ 2.4 log n | 较少 | 宽松 |
| 0.9 | log n / log(10/9) ≈ 6.6 log n | 很少 | 极宽松 |

## 7. 与其他平衡树对比

| 特性 | 替罪羊树 | AVL树 | 红黑树 |
|------|---------|-------|--------|
| 旋转 | 不需要 | 需要 | 需要 |
| 额外信息 | size | height | color |
| 实现难度 | 简单 | 中等 | 复杂 |
| 单次最坏 | O(n) | O(log n) | O(log n) |
| 均摊 | O(log n) | O(log n) | O(log n) |

## 8. 应用场景

1. 简单的有序集合实现
2. 不希望实现旋转操作的场景
3. 对均摊性能要求高但不关心单次最坏
4. 教学目的

## 9. 总结

替罪羊树用重建代替旋转，实现简单，均摊性能好。关键在于选择合适的 alpha 值和正确实现重建操作。适合需要简洁实现且能接受均摊复杂度的场景。
