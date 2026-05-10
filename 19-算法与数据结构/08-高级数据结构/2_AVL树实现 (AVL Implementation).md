# AVL树完整实现 (AVL Implementation)

## 1. 整体结构

本节给出AVL树的完整Python实现，包括插入、删除、查找和遍历操作。

## 2. 节点定义与辅助函数

```python
class AVLNode:
    """AVL树节点"""
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """AVL树完整实现"""

    def get_height(self, node):
        if node is None:
            return 0
        return node.height

    def get_balance(self, node):
        if node is None:
            return 0
        return self.get_height(node.left) - self.get_height(node.right)

    def update_height(self, node):
        node.height = 1 + max(self.get_height(node.left),
                              self.get_height(node.right))
```

## 3. 旋转操作

```python
    def right_rotate(self, y):
        """右旋（LL旋转）"""
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self.update_height(y)
        self.update_height(x)
        return x

    def left_rotate(self, x):
        """左旋（RR旋转）"""
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self.update_height(x)
        self.update_height(y)
        return y
```

## 4. 插入操作

### 4.1 插入逻辑

1. 按照BST规则递归插入
2. 回溯时更新节点高度
3. 检查平衡因子，若不平衡则执行旋转

### 4.2 代码实现

```python
    def insert(self, root, key):
        """插入节点"""
        # 1. 标准BST插入
        if root is None:
            return AVLNode(key)

        if key < root.key:
            root.left = self.insert(root.left, key)
        elif key > root.key:
            root.right = self.insert(root.right, key)
        else:
            return root  # 不允许重复键

        # 2. 更新当前节点高度
        self.update_height(root)

        # 3. 获取平衡因子
        balance = self.get_balance(root)

        # 4. 根据不平衡情况执行旋转
        # LL情况
        if balance > 1 and key < root.left.key:
            return self.right_rotate(root)

        # RR情况
        if balance < -1 and key > root.right.key:
            return self.left_rotate(root)

        # LR情况
        if balance > 1 and key > root.left.key:
            root.left = self.left_rotate(root.left)
            return self.right_rotate(root)

        # RL情况
        if balance < -1 and key < root.right.key:
            root.right = self.right_rotate(root.right)
            return self.left_rotate(root)

        return root
```

## 5. 删除操作

### 5.1 删除逻辑

1. 按照BST规则递归删除
2. 找到要删除的节点后，分三种情况处理
3. 回溯时更新高度并检查平衡

### 5.2 代码实现

```python
    def get_min_node(self, node):
        """获取最小值节点"""
        current = node
        while current.left is not None:
            current = current.left
        return current

    def delete(self, root, key):
        """删除节点"""
        if root is None:
            return root

        if key < root.key:
            root.left = self.delete(root.left, key)
        elif key > root.key:
            root.right = self.delete(root.right, key)
        else:
            # 找到要删除的节点
            if root.left is None:
                return root.right
            elif root.right is None:
                return root.left
            else:
                # 有两个子节点：找中序后继
                successor = self.get_min_node(root.right)
                root.key = successor.key
                root.right = self.delete(root.right, successor.key)

        # 更新高度并检查平衡
        self.update_height(root)
        balance = self.get_balance(root)

        if balance > 1:
            if self.get_balance(root.left) >= 0:
                return self.right_rotate(root)
            else:
                root.left = self.left_rotate(root.left)
                return self.right_rotate(root)

        if balance < -1:
            if self.get_balance(root.right) <= 0:
                return self.left_rotate(root)
            else:
                root.right = self.right_rotate(root.right)
                return self.left_rotate(root)

        return root
```

## 6. 查找操作

```python
    def search(self, root, key):
        """查找节点，时间复杂度 O(log n)"""
        if root is None or root.key == key:
            return root
        if key < root.key:
            return self.search(root.left, key)
        return self.search(root.right, key)

    def find_min(self, root):
        """查找最小值"""
        if root is None:
            return None
        while root.left is not None:
            root = root.left
        return root.key

    def find_max(self, root):
        """查找最大值"""
        if root is None:
            return None
        while root.right is not None:
            root = root.right
        return root.key
```

## 7. 遍历操作

```python
    def inorder(self, root):
        """中序遍历（有序输出）"""
        result = []
        if root:
            result.extend(self.inorder(root.left))
            result.append(root.key)
            result.extend(self.inorder(root.right))
        return result

    def preorder(self, root):
        """前序遍历"""
        result = []
        if root:
            result.append(root.key)
            result.extend(self.preorder(root.left))
            result.extend(self.preorder(root.right))
        return result

    def level_order(self, root):
        """层序遍历"""
        if root is None:
            return []
        result = []
        queue = [root]
        while queue:
            node = queue.pop(0)
            result.append(node.key)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return result
```

## 8. 打印树结构

```python
    def display(self, root, level=0, prefix="Root: "):
        """美观地打印树结构"""
        if root is not None:
            print(" " * (level * 4) + prefix + str(root.key)
                  + f" (h={root.height}, bf={self.get_balance(root)})")
            if root.left or root.right:
                if root.left:
                    self.display(root.left, level + 1, "L--- ")
                if root.right:
                    self.display(root.right, level + 1, "R--- ")
```

## 9. 完整使用示例

```python
if __name__ == "__main__":
    tree = AVLTree()
    root = None

    keys = [10, 20, 30, 40, 50, 25, 5, 6, 7]
    print("AVL树插入过程演示")
    print("=" * 50)

    for key in keys:
        root = tree.insert(root, key)
        print(f"\n插入 {key} 后:")
        tree.display(root)

    print(f"\n中序遍历: {tree.inorder(root)}")
    print(f"前序遍历: {tree.preorder(root)}")
    print(f"最小值: {tree.find_min(root)}")
    print(f"最大值: {tree.find_max(root)}")
    print(f"树高度: {tree.get_height(root)}")

    # 删除演示
    for key in [10, 25]:
        root = tree.delete(root, key)
        print(f"\n删除 {key} 后中序遍历: {tree.inorder(root)}")
```

## 10. C++ 完整实现

```cpp
#include <iostream>
#include <algorithm>
#include <queue>
using namespace std;

struct AVLNode {
    int key;
    AVLNode* left;
    AVLNode* right;
    int height;
    AVLNode(int k) : key(k), left(nullptr), right(nullptr), height(1) {}
};

class AVLTree {
public:
    int getHeight(AVLNode* node) {
        return node ? node->height : 0;
    }

    int getBalance(AVLNode* node) {
        return node ? getHeight(node->left) - getHeight(node->right) : 0;
    }

    void updateHeight(AVLNode* node) {
        node->height = 1 + max(getHeight(node->left), getHeight(node->right));
    }

    AVLNode* rightRotate(AVLNode* y) {
        AVLNode* x = y->left;
        AVLNode* T2 = x->right;
        x->right = y;
        y->left = T2;
        updateHeight(y);
        updateHeight(x);
        return x;
    }

    AVLNode* leftRotate(AVLNode* x) {
        AVLNode* y = x->right;
        AVLNode* T2 = y->left;
        y->left = x;
        x->right = T2;
        updateHeight(x);
        updateHeight(y);
        return y;
    }

    AVLNode* insert(AVLNode* root, int key) {
        if (!root) return new AVLNode(key);
        if (key < root->key)
            root->left = insert(root->left, key);
        else if (key > root->key)
            root->right = insert(root->right, key);
        else return root;

        updateHeight(root);
        int balance = getBalance(root);

        if (balance > 1 && key < root->left->key)
            return rightRotate(root);
        if (balance < -1 && key > root->right->key)
            return leftRotate(root);
        if (balance > 1 && key > root->left->key) {
            root->left = leftRotate(root->left);
            return rightRotate(root);
        }
        if (balance < -1 && key < root->right->key) {
            root->right = rightRotate(root->right);
            return leftRotate(root);
        }
        return root;
    }
};
```

## 11. 时间复杂度总结

| 操作 | 时间复杂度 | 说明 |
|------|-----------|------|
| 插入 | O(log n) | 包含旋转开销 |
| 删除 | O(log n) | 包含旋转开销 |
| 查找 | O(log n) | 与普通BST相同 |
| 遍历 | O(n) | 访问所有节点 |
| 旋转 | O(1) | 单次旋转 |
