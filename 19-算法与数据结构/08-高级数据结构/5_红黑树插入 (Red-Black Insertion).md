# 红黑树插入 (Red-Black Insertion)

## 1. 插入概述

红黑树的插入操作分为两步：
1. 标准BST插入：找到插入位置，插入新节点（默认红色）
2. 修复红黑性质：从新节点开始向上修复，直到满足所有性质

## 2. 插入算法

### 2.1 第一步：BST插入

```python
def insert(self, key):
    """插入新键值"""
    new_node = RBNode(key)
    new_node.left = self.NIL
    new_node.right = self.NIL

    # 标准BST插入
    parent = None
    current = self.root

    while current != self.NIL:
        parent = current
        if key < current.key:
            current = current.left
        elif key > current.key:
            current = current.right
        else:
            return  # 重复键，不插入

    new_node.parent = parent

    if parent is None:
        self.root = new_node
    elif key < parent.key:
        parent.left = new_node
    else:
        parent.right = new_node

    # 如果新节点是根节点，染黑返回
    if new_node.parent is None:
        new_node.color = Color.BLACK
        return

    # 如果祖父节点不存在，无需修复
    if new_node.parent.parent is None:
        return

    # 修复红黑性质
    self._fix_insert(new_node)
```

## 3. 修复的五种情况

设新插入的节点为 N，父节点为 P，祖父节点为 G，叔节点为 U。

### 情况0：N 是根节点
条件：N.parent = None
操作：将 N 染黑

### 情况1：父节点是黑色
条件：P.color = BLACK
操作：无需任何修复。插入红色节点不改变黑高，且不产生连续红色。

### 情况2：父红叔红（变色 + 递归）
条件：P.color = RED 且 U.color = RED
操作：
1. 将 P 染黑
2. 将 U 染黑
3. 将 G 染红
4. 以 G 为新的当前节点，递归修复

```
修复前：           修复后：
      G(B)              G(R) <- 继续向上修复
     / \               / \
   P(R) U(R)  ->     P(B) U(B)
   /                /
  N(R)            N(R)
```

### 情况3：父红叔黑，N是P的左子，P是G的左子（LL）
条件：P.color = RED, U.color = BLACK, N = P.left, P = G.left
操作：
1. 将 P 染黑
2. 将 G 染红
3. 对 G 右旋

```
修复前：           修复后：
      G(B)              P(B)
     / \               / \
   P(R) U(B)  ->     N(R) G(R)
   /                       \
  N(R)                     U(B)
```

### 情况4：父红叔黑，N是P的右子，P是G的左子（LR）
条件：P.color = RED, U.color = BLACK, N = P.right, P = G.left
操作：
1. 对 P 左旋
2. 转化为情况3

## 4. 完整修复代码

```python
def _fix_insert(self, node):
    """修复红黑树插入后的性质违反"""
    while node != self.root and node.parent.color == Color.RED:
        if node.parent == node.parent.parent.left:
            uncle = node.parent.parent.right

            if uncle.color == Color.RED:
                # 情况2：父红叔红 -> 变色
                node.parent.color = Color.BLACK
                uncle.color = Color.BLACK
                node.parent.parent.color = Color.RED
                node = node.parent.parent
            else:
                if node == node.parent.right:
                    # 情况4：先左旋
                    node = node.parent
                    self._left_rotate(node)
                # 情况3：变色 + 右旋
                node.parent.color = Color.BLACK
                node.parent.parent.color = Color.RED
                self._right_rotate(node.parent.parent)
        else:
            # 镜像情况
            uncle = node.parent.parent.left

            if uncle.color == Color.RED:
                node.parent.color = Color.BLACK
                uncle.color = Color.BLACK
                node.parent.parent.color = Color.RED
                node = node.parent.parent
            else:
                if node == node.parent.left:
                    node = node.parent
                    self._right_rotate(node)
                node.parent.color = Color.BLACK
                node.parent.parent.color = Color.RED
                self._left_rotate(node.parent.parent)

    # 确保根节点为黑色
    self.root.color = Color.BLACK
```

## 5. C++ 插入实现

```cpp
void fixInsert(RBNode* node) {
    while (node != root && node->parent->color == RED) {
        if (node->parent == node->parent->parent->left) {
            RBNode* uncle = node->parent->parent->right;

            if (uncle && uncle->color == RED) {
                node->parent->color = BLACK;
                uncle->color = BLACK;
                node->parent->parent->color = RED;
                node = node->parent->parent;
            } else {
                if (node == node->parent->right) {
                    node = node->parent;
                    leftRotate(node);
                }
                node->parent->color = BLACK;
                node->parent->parent->color = RED;
                rightRotate(node->parent->parent);
            }
        } else {
            // 镜像情况类似处理
        }
    }
    root->color = BLACK;
}
```

## 6. 插入修复的最多旋转次数

定理：红黑树插入后，最多需要 2 次旋转即可恢复平衡。

分析：
- 情况2（父红叔红）：只变色不旋转，但可能向上递归
- 情况3：1次旋转
- 情况4：2次旋转（先转化为情况3）
- 递归路径上至多执行一次情况3或情况4

## 7. 插入示例演示

依次插入序列：10, 20, 30, 15, 25

```
插入10:  10(B)
插入20:  10(B)
            \
            20(R)
插入30:  10(B)
            \
            20(R)
              \
              30(R) -> 需要修复 -> 20(B)
                                /    \
                             10(R)  30(R)
插入15:  20(B)
        /    \
     10(R)  30(R)
        \
        15(R)
插入25:  20(B)
        /    \
     10(R)  30(R)
       \      /
       15(R) 25(R) -> 需要修复
```

## 8. 插入操作的复杂度

| 指标 | 复杂度 |
|------|--------|
| 查找插入位置 | O(log n) |
| 插入节点 | O(1) |
| 修复（变色） | O(log n) 最坏 |
| 修复（旋转） | 最多2次，O(1) |
| 总时间 | O(log n) |

## 9. 总结

红黑树的插入修复逻辑清晰：
1. 新节点默认红色，减少需要修复的情况
2. 父红叔红时只需变色，可能递归
3. 父红叔黑时需要旋转，至多2次
4. 最终保证根节点为黑色
