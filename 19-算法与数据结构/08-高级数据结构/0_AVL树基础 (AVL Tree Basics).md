# AVL树基础 (AVL Tree Basics)

## 1. 概述

AVL树是一种**自平衡二叉搜索树**，由苏联数学家 Adelson-Velsky 和 Landis 于 1962 年提出，是最早被发明的自平衡二叉搜索树。

AVL树的核心思想：在每次插入或删除操作后，通过**旋转操作**维持树的平衡，保证任意节点的左右子树高度差不超过1。

## 2. 平衡条件（Balance Condition）

### 2.1 平衡因子（Balance Factor）

对于任意节点 N，其平衡因子定义为：

```
BF(N) = height(N.left) - height(N.right)
```

AVL树要求：每个节点的平衡因子必须属于集合 {-1, 0, 1}。

- BF = 0：左右子树等高
- BF = 1：左子树比右子树高1
- BF = -1：右子树比左子树高1
- |BF| >= 2：不平衡，需要旋转修复

### 2.2 高度约束

对于含有 n 个节点的AVL树：

- 最大高度约为 1.44 * log2(n+2)
- 最小节点数递推公式：N(h) = N(h-1) + N(h-2) + 1，其中 N(0)=1, N(1)=2

高度为 h 的AVL树至少有 F(h+2) - 1 个节点，与斐波那契数列密切相关。

### 2.3 为什么需要平衡？

普通BST在最坏情况下会退化为链表：

| 操作 | 普通BST（最坏） | AVL树 |
|------|----------------|-------|
| 查找 | O(n) | O(log n) |
| 插入 | O(n) | O(log n) |
| 删除 | O(n) | O(log n) |

## 3. 旋转类型总览

当平衡因子变为 +2 或 -2 时，需要通过旋转恢复平衡。旋转分为四种基本类型：

### 3.1 单旋转（Single Rotation）

- LL旋转（右旋）：不平衡节点的左子树的左子树过重
- RR旋转（左旋）：不平衡节点的右子树的右子树过重

### 3.2 双旋转（Double Rotation）

- LR旋转：不平衡节点的左子树的右子树过重
- RL旋转：不平衡节点的右子树的左子树过重

## 4. AVL树节点定义

### 4.1 Python 实现

```python
class AVLNode:
    """AVL树节点"""
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1  # 新节点初始高度为1

    def __repr__(self):
        return f"AVLNode({self.key}, h={self.height})"
```

### 4.2 C++ 实现

```cpp
struct AVLNode {
    int key;
    AVLNode* left;
    AVLNode* right;
    int height;

    AVLNode(int k) : key(k), left(nullptr), right(nullptr), height(1) {}
};
```

## 5. 辅助函数

### 5.1 获取高度

```python
def get_height(node):
    """获取节点高度，空节点高度为0"""
    if node is None:
        return 0
    return node.height
```

### 5.2 获取平衡因子

```python
def get_balance(node):
    """获取节点的平衡因子"""
    if node is None:
        return 0
    return get_height(node.left) - get_height(node.right)
```

### 5.3 更新高度

```python
def update_height(node):
    """更新节点高度"""
    if node is not None:
        node.height = 1 + max(get_height(node.left), get_height(node.right))
```

## 6. AVL树与普通BST的对比

| 特性 | 普通BST | AVL树 |
|------|---------|-------|
| 平衡保证 | 无 | 严格平衡（高度差<=1） |
| 查找效率 | O(n) 最坏 | O(log n) 保证 |
| 插入开销 | O(1) 额外 | 需要旋转，最多O(log n) |
| 删除开销 | O(1) 额外 | 需要旋转，最多O(log n) |
| 空间开销 | 较小 | 每节点多一个height字段 |
| 适用场景 | 数据随机分布 | 频繁查找、需要稳定性能 |

## 7. 应用场景

1. 数据库索引：需要稳定的查找性能
2. 内存中的有序集合：需要频繁的插入、删除和查找
3. 编译器符号表：管理变量和函数名
4. 操作系统调度：进程管理

## 8. 总结

AVL树通过严格的平衡条件保证了高效的查找性能。虽然插入和删除时可能需要旋转操作来维护平衡，但这些操作的时间复杂度仍然是 O(log n)。AVL树适合查找操作远多于修改操作的场景。

在后续章节中，我们将详细学习四种旋转操作的具体实现。
