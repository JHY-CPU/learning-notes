# 红黑树基础 (Red-Black Tree Basics)

## 1. 概述

红黑树是一种**自平衡二叉搜索树**，由 Rudolf Bayer 于 1972 年发明。它通过对每个节点赋予颜色（红色或黑色），并遵循五条性质来保证树的近似平衡。

红黑树是工程中应用最广泛的自平衡BST：
- C++ STL 中的 std::map、std::set
- Java 中的 TreeMap、TreeSet、HashMap（冲突链表转红黑树）
- Linux 内核的 CFS 调度器、内存管理

## 2. 五条性质（Red-Black Properties）

### 性质1：节点着色
每个节点要么是红色，要么是黑色。

### 性质2：根节点黑色
根节点必须是黑色。

### 性质3：叶子节点黑色
所有叶子节点（NIL节点/空节点）都是黑色。这里的叶子指的是虚拟的NIL节点。

### 性质4：红色节点约束
如果一个节点是红色的，则它的两个子节点都是黑色的。这意味着不能有两个连续的红色节点。

### 性质5：黑色高度一致
从任意节点到其每个叶子节点的所有路径都包含相同数目的黑色节点。这个数目称为黑高（Black Height）。

## 3. 红黑树的平衡保证

### 3.1 高度约束

定理：含有 n 个内部节点的红黑树的高度至多为 2*log2(n+1)。

证明思路：
- 由性质4和性质5，最短路径全黑，最长路径红黑交替
- 最长路径长度不超过最短路径的2倍
- 因此树高为 O(log n)

### 3.2 与AVL树的对比

| 特性 | 红黑树 | AVL树 |
|------|--------|-------|
| 平衡程度 | 近似平衡 | 严格平衡 |
| 树高 | <= 2log(n+1) | <= 1.44log(n+2) |
| 查找 | 略慢 | 略快 |
| 插入旋转 | 最多2次 | 最多1次 |
| 删除旋转 | 最多3次 | 最多O(log n)次 |
| 颜色存储 | 每节点1bit | 无 |
| 高度存储 | 无 | 每节点int |
| 综合性能 | 插入删除更优 | 查找更优 |

## 4. 节点定义

### 4.1 Python 实现

```python
class Color:
    RED = "RED"
    BLACK = "BLACK"

class RBNode:
    """红黑树节点"""
    def __init__(self, key, color=Color.RED):
        self.key = key
        self.color = color
        self.left = None
        self.right = None
        self.parent = None

    def __repr__(self):
        return f"RBNode({self.key}, {self.color})"
```

### 4.2 C++ 实现

```cpp
enum Color { RED, BLACK };

struct RBNode {
    int key;
    Color color;
    RBNode* left;
    RBNode* right;
    RBNode* parent;

    RBNode(int k) : key(k), color(RED),
                    left(nullptr), right(nullptr), parent(nullptr) {}
};
```

### 4.3 为什么新节点默认红色？

新插入的节点默认设为红色，原因如下：
- 红色节点不影响性质5（黑高一致）
- 如果父节点是黑色，则插入后无需调整
- 只有当父节点也是红色时才违反性质4，需要修复
- 这样可以减少需要修复的情况

## 5. NIL哨兵节点

在实际实现中，所有 None 指针可以用一个共享的 NIL 哨兵节点代替：

```python
NIL = RBNode(None)
NIL.color = Color.BLACK
NIL.left = NIL.right = NIL.parent = NIL
```

优点：
- 避免空指针检查
- 所有叶子天然为黑色（满足性质3）
- 简化代码逻辑

## 6. 红黑树的基本操作框架

```python
class RedBlackTree:
    def __init__(self):
        self.NIL = RBNode(None, Color.BLACK)
        self.NIL.left = self.NIL.right = self.NIL
        self.root = self.NIL

    def is_red(self, node):
        if node is None or node == self.NIL:
            return False
        return node.color == Color.RED

    def is_black(self, node):
        return not self.is_red(node)

    def black_height(self, node):
        if node == self.NIL:
            return 1
        bh = self.black_height(node.left)
        if self.is_black(node):
            bh += 1
        return bh
```

## 7. 红黑树的旋转操作

红黑树的旋转与AVL树类似，但需要额外维护 parent 指针：

```python
def left_rotate(self, x):
    """左旋"""
    y = x.right
    x.right = y.left

    if y.left != self.NIL:
        y.left.parent = x

    y.parent = x.parent

    if x.parent is None:
        self.root = y
    elif x == x.parent.left:
        x.parent.left = y
    else:
        x.parent.right = y

    y.left = x
    x.parent = y
```

## 8. 应用场景

1. 关联容器实现：C++ map/set，Java TreeMap/TreeSet
2. 进程调度：Linux CFS 使用红黑树管理进程
3. 内存管理：虚拟内存区域（VMA）的管理
4. 定时器管理：内核定时器的组织
5. IO多路复用：epoll 中就绪事件的管理

## 9. 时间复杂度

| 操作 | 时间复杂度 |
|------|-----------|
| 查找 | O(log n) |
| 插入 | O(log n) |
| 删除 | O(log n) |
| 最大/最小 | O(log n) |

## 10. 总结

红黑树通过五条颜色性质保证了树的近似平衡。相比AVL树，删除操作的旋转次数更少，插入删除的综合性能更优，因此在工程实践中被广泛采用。
