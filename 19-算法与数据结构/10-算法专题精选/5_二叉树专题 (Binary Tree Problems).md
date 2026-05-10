# 二叉树专题 (Binary Tree Problems)

## 一、概念定义与原理

### 1.1 二叉树基本概念

**二叉树：** 每个节点最多有两个子节点（左子和右子）。

**特殊二叉树：**
- **满二叉树：** 每层都是满的
- **完全二叉树：** 除最后一层外都是满的，最后一层从左到右填充
- **二叉搜索树 (BST)：** 左子树所有节点 < 根 < 右子树所有节点
- **平衡二叉树：** 左右子树高度差不超过1

### 1.2 三种遍历

- **前序遍历：** 根-左-右
- **中序遍历：** 左-根-右
- **后序遍历：** 左-右-根

### 1.3 LCA（最近公共祖先）

给定两节点，找到它们最近的公共祖先。

---

## 二、核心算法

### 2.1 遍历（递归与迭代）

递归天然适合树的遍历。迭代版本用栈模拟。

### 2.2 层序遍历

用队列 BFS，逐层处理。

### 2.3 LCA

- **递归法：** 若当前节点是其中一个目标或空，返回当前节点。递归左右子树，若都有结果则当前节点为 LCA。
- **倍增法：** 预处理祖先表，$O(\log n)$ 查询。

---

## 三、代码实现

### 3.1 树节点与遍历 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x): val(x), left(nullptr), right(nullptr) {}
};

// 前序遍历（递归）
void preorder(TreeNode* root, vector<int>& result) {
    if (!root) return;
    result.push_back(root->val);
    preorder(root->left, result);
    preorder(root->right, result);
}

// 前序遍历（迭代）
vector<int> preorder_iter(TreeNode* root) {
    vector<int> result;
    if (!root) return result;
    stack<TreeNode*> stk;
    stk.push(root);
    while (!stk.empty()) {
        TreeNode* node = stk.top(); stk.pop();
        result.push_back(node->val);
        if (node->right) stk.push(node->right);
        if (node->left) stk.push(node->left);
    }
    return result;
}

// 中序遍历（迭代）
vector<int> inorder_iter(TreeNode* root) {
    vector<int> result;
    stack<TreeNode*> stk;
    TreeNode* curr = root;
    while (curr || !stk.empty()) {
        while (curr) { stk.push(curr); curr = curr->left; }
        curr = stk.top(); stk.pop();
        result.push_back(curr->val);
        curr = curr->right;
    }
    return result;
}
```

### 3.2 层序遍历 - C++

```cpp
vector<vector<int>> level_order(TreeNode* root) {
    vector<vector<int>> result;
    if (!root) return result;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int size = q.size();
        vector<int> level;
        for (int i = 0; i < size; i++) {
            TreeNode* node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left) q.push(node->left);
            if (node->right) q.push(node->right);
        }
        result.push_back(level);
    }
    return result;
}
```

### 3.3 LCA - C++

```cpp
// 递归法 LCA（适用于二叉树）
TreeNode* lowest_common_ancestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    if (!root || root == p || root == q) return root;
    TreeNode* left = lowest_common_ancestor(root->left, p, q);
    TreeNode* right = lowest_common_ancestor(root->right, p, q);
    if (left && right) return root;
    return left ? left : right;
}
```

### 3.4 序列化与反序列化

```cpp
// 前序序列化
string serialize(TreeNode* root) {
    if (!root) return "#";
    return to_string(root->val) + "," + serialize(root->left) + "," + serialize(root->right);
}

TreeNode* deserialize_helper(queue<string>& q) {
    string val = q.front(); q.pop();
    if (val == "#") return nullptr;
    TreeNode* node = new TreeNode(stoi(val));
    node->left = deserialize_helper(q);
    node->right = deserialize_helper(q);
    return node;
}
```

### 3.5 Python 实现

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val; self.left = left; self.right = right

def level_order(root):
    if not root: return []
    result = []; q = deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        result.append(level)
    return result

def lca(root, p, q):
    if not root or root == p or root == q: return root
    left = lca(root.left, p, q)
    right = lca(root.right, p, q)
    return root if left and right else (left or right)

def inorder(root):
    result = []; stk = []; curr = root
    while curr or stk:
        while curr: stk.append(curr); curr = curr.left
        curr = stk.pop(); result.append(curr.val)
        curr = curr.right
    return result
```

---

## 四、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 遍历 | $O(n)$ | $O(h)$ 栈空间 |
| 层序遍历 | $O(n)$ | $O(w)$ 最大宽度 |
| LCA（递归） | $O(n)$ | $O(h)$ |
| LCA（倍增） | $O(n \log n)$ 预处理，$O(\log n)$ 查询 | $O(n \log n)$ |
| 序列化 | $O(n)$ | $O(n)$ |

---

## 五、竞赛与面试应用场景

1. **LeetCode 94/144/145：** 三种遍历
2. **LeetCode 236：** 二叉树的最近公共祖先
3. **LeetCode 297：** 二叉树的序列化与反序列化
4. **LeetCode 102：** 二叉树的层序遍历
5. **LeetCode 124：** 二叉树中的最大路径和
