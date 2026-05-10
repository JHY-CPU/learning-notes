# 红黑树删除 (Red-Black Deletion)

## 1. 删除概述

红黑树的删除是所有操作中最复杂的部分。删除过程分为三步：
1. 标准BST删除：找到并移除目标节点
2. 处理颜色：如果删除的是黑色节点，可能破坏性质5
3. 修复红黑性质：通过变色和旋转恢复平衡

## 2. BST删除回顾

删除节点 z 时有三种情况：
- 无子节点：直接删除
- 只有一个子节点：用子节点替代
- 有两个子节点：找中序后继替代，然后删除后继

关键：实际上被删除的节点 y 最多只有一个非NIL子节点。

## 3. 删除算法

```python
def delete(self, key):
    """删除指定键值的节点"""
    z = self._search(key)
    if z == self.NIL:
        return

    y = z
    y_original_color = y.color

    if z.left == self.NIL:
        x = z.right
        self._transplant(z, z.right)
    elif z.right == self.NIL:
        x = z.left
        self._transplant(z, z.left)
    else:
        y = self._minimum(z.right)
        y_original_color = y.color
        x = y.right

        if y.parent == z:
            x.parent = y
        else:
            self._transplant(y, y.right)
            y.right = z.right
            y.right.parent = y

        self._transplant(z, y)
        y.left = z.left
        y.left.parent = y
        y.color = z.color

    if y_original_color == Color.BLACK:
        self._fix_delete(x)

def _transplant(self, u, v):
    """用v替换u在树中的位置"""
    if u.parent is None:
        self.root = v
    elif u == u.parent.left:
        u.parent.left = v
    else:
        u.parent.right = v
    v.parent = u.parent
```

## 4. 删除修复详解

删除黑色节点后，被替代的节点 x 相对于原来多了一层"黑色"，称之为双黑（Double Black）。

修复的核心：消除双黑，使每条路径的黑高恢复一致。

设 x 的兄弟为 S，父节点为 P。

### 情况1：x 是根节点
操作：直接移除双黑标记。根节点多一层黑色不影响性质。

### 情况2：兄弟红
条件：S.color = RED
操作：
1. 将 S 染黑，P 染红
2. 对 P 做旋转
3. 更新兄弟节点，转化为其他情况

### 情况3：兄弟黑，两个侄子都黑
条件：S.color = BLACK, Sl.color = BLACK, Sr.color = BLACK
操作：
1. 将 S 染红
2. 将 x 的双黑上移到 P
3. 以 P 为新的 x，继续修复

### 情况4：兄弟黑，近侄子红，远侄子黑
条件：S.color = BLACK, 近侄子为红，远侄子为黑
操作：
1. 将近侄子染黑，S 染红
2. 对 S 做旋转
3. 转化为情况5

### 情况5：兄弟黑，远侄子红
条件：S.color = BLACK, 远侄子为红
操作：
1. S 染成 P 的颜色
2. P 染黑
3. 远侄子染黑
4. 对 P 做旋转
5. 移除双黑，修复完成

## 5. 完整删除修复代码

```python
def _fix_delete(self, x):
    """修复删除后的红黑性质"""
    while x != self.root and x.color == Color.BLACK:
        if x == x.parent.left:
            S = x.parent.right

            # 情况2：兄弟红
            if S.color == Color.RED:
                S.color = Color.BLACK
                x.parent.color = Color.RED
                self._left_rotate(x.parent)
                S = x.parent.right

            # 情况3：兄弟黑，两个侄子都黑
            if S.left.color == Color.BLACK and S.right.color == Color.BLACK:
                S.color = Color.RED
                x = x.parent
            else:
                # 情况4：远侄子黑
                if S.right.color == Color.BLACK:
                    S.left.color = Color.BLACK
                    S.color = Color.RED
                    self._right_rotate(S)
                    S = x.parent.right

                # 情况5：远侄子红
                S.color = x.parent.color
                x.parent.color = Color.BLACK
                S.right.color = Color.BLACK
                self._left_rotate(x.parent)
                x = self.root
        else:
            # 镜像情况
            S = x.parent.left
            # ... 镜像代码类似
    x.color = Color.BLACK
```

## 6. 删除操作的旋转次数

定理：红黑树删除后，最多需要 3 次旋转。

- 情况2：1次旋转 + 转化为其他情况
- 情况4+5：2次旋转
- 情况3：0次旋转，但可能向上递归

相比AVL树删除可能需要 O(log n) 次旋转，红黑树删除更高效。

## 7. 复杂度分析

| 操作 | 时间复杂度 |
|------|-----------|
| 查找删除目标 | O(log n) |
| 找中序后继 | O(log n) |
| 修复 | O(log n) |
| 旋转 | 最多3次 |
| 总时间 | O(log n) |

## 8. 与AVL树删除的对比

| 特性 | 红黑树 | AVL树 |
|------|--------|-------|
| 最大旋转次数 | 3次 | O(log n)次 |
| 修复复杂度 | 较复杂（5种情况） | 较简单（4种情况） |
| 总体删除效率 | 更优 | 略慢 |

## 9. 总结

红黑树删除修复的核心逻辑：
1. 删除黑色节点会破坏性质5（黑高一致性）
2. 通过兄弟节点的颜色和侄子节点的情况分类处理
3. 本质是将"双黑"向上传递或就地消除
4. 最多需要3次旋转即可恢复平衡
