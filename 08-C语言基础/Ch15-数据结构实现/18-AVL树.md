# AVL树

## 1. 概述

AVL树（Adelson-Velsky and Landis Tree）是**自平衡二叉搜索树**。它在每个节点上维护一个**平衡因子**（Balance Factor），并在插入或删除后通过**旋转**操作来保持平衡。

### 为什么需要 AVL 树

普通 BST 在最坏情况下会退化为链表（O(n)）。AVL树保证树的高度始终为 O(log n)。

## 2. 平衡因子

**平衡因子（Balance Factor）** = 左子树高度 - 右子树高度

### AVL 树的性质
- 任意节点的平衡因子只能是 **-1, 0, 1**
- 如果平衡因子绝对值超过 1，则需要通过旋转来调整

```
平衡因子:  BF = height(left) - height(right)

BF =  1:  左子树比右子树高一层
BF =  0:  左右子树等高
BF = -1:  右子树比左子树高一层
```

## 3. 节点定义

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct AVLNode {
    int data;
    int height;             // 节点高度
    struct AVLNode *left;
    struct AVLNode *right;
} AVLNode;

// 获取节点高度（NULL高度为0）
int getHeight(AVLNode *node) {
    return node ? node->height : 0;
}

// 获取平衡因子
int getBalanceFactor(AVLNode *node) {
    if (!node) return 0;
    return getHeight(node->left) - getHeight(node->right);
}

// 更新节点高度
void updateHeight(AVLNode *node) {
    int lh = getHeight(node->left);
    int rh = getHeight(node->right);
    node->height = (lh > rh ? lh : rh) + 1;
}

// 创建节点
AVLNode *createAVLNode(int value) {
    AVLNode *node = (AVLNode *)malloc(sizeof(AVLNode));
    node->data = value;
    node->height = 1;
    node->left = NULL;
    node->right = NULL;
    return node;
}
```

## 4. 四种旋转操作

### 4.1 LL 右旋转（右单旋）

当在**左子树的左子树**插入导致不平衡时，进行右旋转。

```
    y               x
   / \             / \
  x   C    ->     A   y
 / \                 / \
A   B               B   C
```

```c
// 右旋转（LL）
AVLNode *rotateRight(AVLNode *y) {
    AVLNode *x = y->left;
    AVLNode *B = x->right;

    x->right = y;
    y->left = B;

    updateHeight(y);
    updateHeight(x);

    return x;  // x 成为新的根
}
```

### 4.2 RR 左旋转（左单旋）

当在**右子树的右子树**插入导致不平衡时，进行左旋转。

```
  x                   y
 / \                 / \
A   y      ->       x   C
   / \             / \
  B   C           A   B
```

```c
// 左旋转（RR）
AVLNode *rotateLeft(AVLNode *x) {
    AVLNode *y = x->right;
    AVLNode *B = y->left;

    y->left = x;
    x->right = B;

    updateHeight(x);
    updateHeight(y);

    return y;
}
```

### 4.3 LR 先左后右（双旋）

当在**左子树的右子树**插入导致不平衡时：
1. 先对左子树进行左旋转
2. 再对当前节点进行右旋转

```
    z               z               y
   / \             / \             / \
  x   D   ->      y   D    ->    x   z
 / \             / \             / \ / \
A   y           x   C           A  B C  D
   / \         / \
  B   C       A   B
```

```c
// LR 双旋转
AVLNode *rotateLR(AVLNode *node) {
    node->left = rotateLeft(node->left);
    return rotateRight(node);
}
```

### 4.4 RL 先右后左（双旋）

当在**右子树的左子树**插入导致不平衡时：
1. 先对右子树进行右旋转
2. 再对当前节点进行左旋转

```c
// RL 双旋转
AVLNode *rotateRL(AVLNode *node) {
    node->right = rotateRight(node->right);
    return rotateLeft(node);
}
```

## 5. 插入操作

```c
AVLNode *insert(AVLNode *root, int value) {
    // 1. 正常 BST 插入
    if (root == NULL) return createAVLNode(value);

    if (value < root->data) {
        root->left = insert(root->left, value);
    } else if (value > root->data) {
        root->right = insert(root->right, value);
    } else {
        return root;  // 不允许重复
    }

    // 2. 更新高度
    updateHeight(root);

    // 3. 获取平衡因子
    int bf = getBalanceFactor(root);

    // 4. 不平衡时进行旋转
    // LL 型：左子树的左子树导致不平衡
    if (bf > 1 && value < root->left->data) {
        return rotateRight(root);
    }
    // RR 型：右子树的右子树导致不平衡
    if (bf < -1 && value > root->right->data) {
        return rotateLeft(root);
    }
    // LR 型：左子树的右子树导致不平衡
    if (bf > 1 && value > root->left->data) {
        return rotateLR(root);
    }
    // RL 型：右子树的左子树导致不平衡
    if (bf < -1 && value < root->right->data) {
        return rotateRL(root);
    }

    return root;
}
```

## 6. 删除操作

```c
AVLNode *delete(AVLNode *root, int value) {
    if (!root) return NULL;

    if (value < root->data) {
        root->left = delete(root->left, value);
    } else if (value > root->data) {
        root->right = delete(root->right, value);
    } else {
        // 找到要删除的节点
        if (!root->left || !root->right) {
            AVLNode *temp = root->left ? root->left : root->right;
            free(root);
            return temp;
        }
        // 有两个子节点
        AVLNode *successor = root->right;
        while (successor->left) successor = successor->left;
        root->data = successor->data;
        root->right = delete(root->right, successor->data);
    }

    updateHeight(root);
    int bf = getBalanceFactor(root);

    // 重新平衡
    if (bf > 1 && getBalanceFactor(root->left) >= 0)
        return rotateRight(root);
    if (bf > 1 && getBalanceFactor(root->left) < 0)
        return rotateLR(root);
    if (bf < -1 && getBalanceFactor(root->right) <= 0)
        return rotateLeft(root);
    if (bf < -1 && getBalanceFactor(root->right) > 0)
        return rotateRL(root);

    return root;
}
```

## 7. 中序遍历与验证

```c
void inorder(AVLNode *root) {
    if (!root) return;
    inorder(root->left);
    printf("%d(h=%d,bf=%d) ", root->data, root->height,
           getBalanceFactor(root));
    inorder(root->right);
}

void printTree(AVLNode *root, int depth) {
    if (!root) return;
    printTree(root->right, depth + 1);
    for (int i = 0; i < depth; i++) printf("    ");
    printf("%d\n", root->data);
    printTree(root->left, depth + 1);
}
```

## 8. 完整测试

```c
int main() {
    AVLNode *root = NULL;
    int data[] = {10, 20, 30, 25, 28, 27, 5, 3};
    int n = sizeof(data) / sizeof(data[0]);

    for (int i = 0; i < n; i++) {
        root = insert(root, data[i]);
        printf("插入 %d 后:\n", data[i]);
        printTree(root, 0);
        printf("\n");
    }

    printf("中序遍历: ");
    inorder(root);
    printf("\n");

    return 0;
}
```

## 9. 复杂度分析

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 查找 | O(log n) | 高度始终平衡 |
| 插入 | O(log n) | 查找O(log n) + 最多1次旋转O(1) |
| 删除 | O(log n) | 查找O(log n) + 最多O(log n)次旋转 |

## 10. 重点总结

- AVL 树通过平衡因子（-1, 0, 1）维护平衡
- 四种旋转：LL(右旋)、RR(左旋)、LR(先左后右)、RL(先右后左)
- 插入后最多旋转一次即可恢复平衡
- 删除后可能需要多次旋转（沿路径向上）
- AVL 树的查找效率比普通 BST 更稳定
- 红黑树在实际应用中更常见（插入删除旋转次数更少）

> **记忆技巧**：LL 和 RR 是单旋（"方向一致"），LR 和 RL 是双旋（"方向不一致"，先小旋再大旋）。
