# 二叉树专题 (Binary Tree Problems)

## 一、概念定义与原理

### 1.1 二叉树基本概念

**二叉树：** 每个节点最多有两个子节点（左子和右子）。

**特殊二叉树：**
- **满二叉树：** 每层节点数达到最大值
- **完全二叉树：** 除最后一层外都是满的，最后一层从左到右填充
- **二叉搜索树 (BST)：** 左 < 根 < 右
- **平衡二叉树 (AVL)：** 左右子树高度差不超过1

### 1.2 遍历方式

| 遍历 | 顺序 | 应用 |
|------|------|------|
| 前序 | 根-左-右 | 序列化、复制树 |
| 中序 | 左-根-右 | BST有序输出 |
| 后序 | 左-右-根 | 释放内存、计算树大小 |
| 层序 | 逐层从左到右 | 最短路径、最大宽度 |

---

## 二、核心算法

### 2.1 递归遍历

```python
def preorder(root):
    if not root: return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    if not root: return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    if not root: return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### 2.2 迭代遍历

```python
def inorder_iterative(root):
    result, stack, curr = [], [], root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        result.append(curr.val)
        curr = curr.right
    return result
```

### 2.3 层序遍历 (BFS)

```python
from collections import deque

def level_order(root):
    if not root: return []
    result, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        result.append(level)
    return result
```

---

## 三、经典题目详解

### 3.1 二叉树的最近公共祖先 (LeetCode 236)

```python
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right: return root
    return left or right
```

### 3.2 二叉树的序列化与反序列化 (LeetCode 297)

```python
class Codec:
    def serialize(self, root):
        if not root: return "#"
        return (str(root.val) + ","
                + self.serialize(root.left) + ","
                + self.serialize(root.right))

    def deserialize(self, data):
        vals = iter(data.split(","))
        def build():
            val = next(vals)
            if val == "#": return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node
        return build()
```

### 3.3 二叉树的直径 (LeetCode 543)

```python
def diameter_of_binary_tree(root):
    ans = 0
    def depth(node):
        nonlocal ans
        if not node: return 0
        left = depth(node.left)
        right = depth(node.right)
        ans = max(ans, left + right)
        return max(left, right) + 1
    depth(root)
    return ans
```

### 3.4 验证二叉搜索树 (LeetCode 98)

```python
def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root: return True
    if root.val <= lo or root.val >= hi:
        return False
    return (is_valid_bst(root.left, lo, root.val)
        and is_valid_bst(root.right, root.val, hi))
```

### 3.5 二叉树的最大路径和 (LeetCode 124)

```cpp
int maxPathSum(TreeNode* root) {
    int result = INT_MIN;
    function<int(TreeNode*)> dfs = [&](TreeNode* node) -> int {
        if (!node) return 0;
        int left = max(0, dfs(node->left));
        int right = max(0, dfs(node->right));
        result = max(result, left + right + node->val);
        return node->val + max(left, right);
    };
    dfs(root);
    return result;
}
```

### 3.6 从前序与中序遍历序列构造二叉树 (LeetCode 105)

```python
def build_tree(preorder, inorder):
    if not preorder: return None
    root_val = preorder[0]
    root = TreeNode(root_val)
    mid = inorder.index(root_val)
    root.left = build_tree(preorder[1:mid+1], inorder[:mid])
    root.right = build_tree(preorder[mid+1:], inorder[mid+1:])
    return root
```

---

## 四、BST专题

### 4.1 BST的中序后继 (LeetCode 285)

```python
def inorder_successor(root, p):
    successor = None
    while root:
        if p.val < root.val:
            successor = root
            root = root.left
        else:
            root = root.right
    return successor
```

### 4.2 BST转有序双向链表 (LeetCode 426)

```python
def tree_to_doubly_list(root):
    if not root: return None
    first, last = None, None

    def dfs(node):
        nonlocal first, last
        if not node: return
        dfs(node.left)
        if last:
            last.right = node
            node.left = last
        else:
            first = node
        last = node
        dfs(node.right)

    dfs(root)
    first.left = last
    last.right = first
    return first
```

---

## 五、复杂度分析

| 操作 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 遍历 | $O(n)$ | $O(h)$ 栈空间 |
| 层序遍历 | $O(n)$ | $O(w)$ 最大宽度 |
| LCA（递归） | $O(n)$ | $O(h)$ |
| 序列化 | $O(n)$ | $O(n)$ |
| 验证BST | $O(n)$ | $O(h)$ |
| 构造树 | $O(n)$ | $O(n)$ |

其中 $h$ 为树高，平衡树 $h = O(\log n)$，退化为链表 $h = O(n)$。

---

## 六、面试高频题

1. **LeetCode 94/144/145：** 三种遍历
2. **LeetCode 102：** 二叉树的层序遍历
3. **LeetCode 236：** 最近公共祖先
4. **LeetCode 124：** 二叉树的最大路径和
5. **LeetCode 98：** 验证二叉搜索树
6. **LeetCode 105：** 从前序与中序构造二叉树
7. **LeetCode 543：** 二叉树的直径
8. **LeetCode 297：** 序列化与反序列化
9. **LeetCode 104/111：** 最大/最小深度
10. **LeetCode 208：** 实现Trie（前缀树）
