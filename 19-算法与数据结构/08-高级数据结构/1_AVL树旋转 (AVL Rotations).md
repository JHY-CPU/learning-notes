# AVL树旋转 (AVL Rotations)

## 1. 旋转概述

旋转是AVL树维护平衡的核心操作。当某个节点的平衡因子变为 +2 或 -2 时，需要通过旋转来恢复平衡。旋转操作的时间复杂度为 O(1)。

## 2. LL旋转（右旋）

### 2.1 触发条件

当节点 A 的左子树的左子树导致不平衡时（BF(A) = 2 且 BF(A.left) >= 0），需要进行右旋。

### 2.2 旋转过程

```
        A (BF=2)               B
       / \                    / \
      B   T3      右旋       C   A
     / \         ====>      /   / \
    C   T2                T1   T2  T3
   /
  T1
```

### 2.3 代码实现

```python
def right_rotate(y):
    """对节点y执行右旋操作，返回旋转后的新根节点x"""
    x = y.left
    T2 = x.right

    # 执行旋转
    x.right = y
    y.left = T2

    # 更新高度（先更新y，再更新x）
    update_height(y)
    update_height(x)

    return x
```

```cpp
AVLNode* rightRotate(AVLNode* y) {
    AVLNode* x = y->left;
    AVLNode* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = 1 + max(getHeight(y->left), getHeight(y->right));
    x->height = 1 + max(getHeight(x->left), getHeight(x->right));

    return x;
}
```

## 3. RR旋转（左旋）

### 3.1 触发条件

当节点 A 的右子树的右子树导致不平衡时（BF(A) = -2 且 BF(A.right) <= 0），需要进行左旋。

### 3.2 旋转过程

```
    A (BF=-2)                   B
   / \                        / \
  T1  B          左旋        A   C
     / \        ====>      / \   \
    T2  C                T1  T2  T3
         \
          T3
```

### 3.3 代码实现

```python
def left_rotate(x):
    """对节点x执行左旋操作，返回旋转后的新根节点y"""
    y = x.right
    T2 = y.left

    # 执行旋转
    y.left = x
    x.right = T2

    # 更新高度
    update_height(x)
    update_height(y)

    return y
```

```cpp
AVLNode* leftRotate(AVLNode* x) {
    AVLNode* y = x->right;
    AVLNode* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = 1 + max(getHeight(x->left), getHeight(x->right));
    y->height = 1 + max(getHeight(y->left), getHeight(y->right));

    return y;
}
```

## 4. LR旋转（先左旋后右旋）

### 4.1 触发条件

当节点 A 的左子树的右子树导致不平衡时（BF(A) = 2 且 BF(A.left) = -1），需要先对左子节点左旋，再对自身右旋。

### 4.2 旋转过程

```
      A (BF=2)           A              C
     / \                / \            / \
    B   T4   左旋B     C   T4  右旋A  B   A
   / \        ====>   / \      ====> /   / \
  T1  C              B   T3        T1  T2  T4
     / \            / \
    T2  T3        T1  T2
```

### 4.3 代码实现

```python
def left_right_rotate(node):
    """LR旋转：先对左子节点执行左旋，再对当前节点执行右旋"""
    node.left = left_rotate(node.left)
    return right_rotate(node)
```

## 5. RL旋转（先右旋后左旋）

### 5.1 触发条件

当节点 A 的右子树的左子树导致不平衡时（BF(A) = -2 且 BF(A.right) = 1），需要先对右子节点右旋，再对自身左旋。

### 5.2 旋转过程

```
    A (BF=-2)         A                   C
   / \               / \                 / \
  T1  B   右旋B     T1  C    左旋A     A   B
     / \   ====>       / \    ====>   / \   \
    C   T4            T2  B         T1  T2  T4
   / \                   / \
  T2  T3               T3  T4
```

### 5.3 代码实现

```python
def right_left_rotate(node):
    """RL旋转：先对右子节点执行右旋，再对当前节点执行左旋"""
    node.right = right_rotate(node.right)
    return left_rotate(node)
```

## 6. 旋转选择逻辑

```python
def rebalance(node):
    """根据平衡因子选择合适的旋转方式"""
    balance = get_balance(node)

    # 左子树过重
    if balance > 1:
        if get_balance(node.left) >= 0:
            return right_rotate(node)   # LL情况
        else:
            return left_right_rotate(node)  # LR情况

    # 右子树过重
    if balance < -1:
        if get_balance(node.right) <= 0:
            return left_rotate(node)    # RR情况
        else:
            return right_left_rotate(node)  # RL情况

    return node
```

## 7. 旋转的时间与空间复杂度

| 旋转类型 | 时间复杂度 | 空间复杂度 | 涉及修改指针数 |
|----------|-----------|-----------|--------------|
| LL（右旋）| O(1) | O(1) | 2 |
| RR（左旋）| O(1) | O(1) | 2 |
| LR | O(1) | O(1) | 4 |
| RL | O(1) | O(1) | 4 |

## 8. 旋转的性质

1. 旋转不破坏BST性质：中序遍历结果不变
2. 旋转减少树的高度：旋转后子树高度恢复到插入/删除前
3. 旋转是局部操作：只修改O(1)个指针
4. 至多一次旋转：插入后至多一次单旋转或双旋转即可恢复平衡
5. 删除后可能需要多次旋转：删除可能导致旋转向上传播

## 9. 总结

四种旋转操作覆盖了AVL树所有不平衡情况。掌握旋转的关键是理解每种情况的触发条件和旋转过程。实际编程中，只需要判断平衡因子的正负和子节点的平衡因子符号，即可确定使用哪种旋转。
