# 模拟面试-树 (Mock Interview - Tree)

## 一、面试流程模拟

**时间：** 45分钟
**重点：** 递归思维、树的遍历、BST性质

---

## 二、题目1：二叉树的最近公共祖先 (LeetCode 236, Medium, 15分钟)

### 面试过程

**面试官：** "给定二叉树和两个节点 p, q，找到它们的最近公共祖先。"

**候选人：**
"递归思路：后序遍历，自底向上汇报。

递归逻辑：
- 如果当前节点是 p 或 q 或空，返回当前节点
- 递归左右子树
- 如果左右子树都返回了结果，说明 p 和 q 分别在两侧，当前节点就是 LCA
- 如果只有一侧有结果，返回那一侧"

### 代码

```python
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    if left and right:
        return root
    return left or right
```

**面试官追问：** "如果 p 不在树中怎么办？"
"这个解法假设 p 和 q 一定在树中。如果不在，需要额外验证。可以先遍历确认两者都存在，或者修改递归返回值包含 '是否找到' 的信息。"

**复杂度：** 时间 $O(n)$，空间 $O(h)$（递归栈）。

---

## 三、题目2：二叉树的序列化与反序列化 (LeetCode 297, Hard, 20分钟)

### 面试过程

**候选人：**
"前序遍历序列化。用特殊标记表示空节点（如 '#'），值之间用逗号分隔。反序列化时按相同顺序重建。"

### 代码

```python
class Codec:
    def serialize(self, root):
        if not root:
            return "#"
        return (str(root.val) + ","
                + self.serialize(root.left) + ","
                + self.serialize(root.right))

    def deserialize(self, data):
        tokens = iter(data.split(","))

        def build():
            val = next(tokens)
            if val == "#":
                return None
            node = TreeNode(int(val))
            node.left = build()
            node.right = build()
            return node

        return build()
```

**面试官追问：** "层序遍历序列化呢？"
"层序遍历也可以，用队列 BFS。前序的优点是递归简洁，层序的优点是序列化的字符串对人类更可读。"

---

## 四、题目3：验证二叉搜索树 (LeetCode 98, Medium, 10分钟)

### 面试过程

**候选人：**
"常见错误是只比较左右子节点和根的关系。正确做法是维护整个子树的值范围。

方法1：递归传递上下界
方法2：中序遍历检查是否严格递增"

### 代码

```python
# 方法1：范围递归
def is_valid_bst(root, lo=float('-inf'), hi=float('inf')):
    if not root:
        return True
    if root.val <= lo or root.val >= hi:
        return False
    return (is_valid_bst(root.left, lo, root.val)
        and is_valid_bst(root.right, root.val, hi))

# 方法2：中序遍历
def is_valid_bst_inorder(root):
    prev = float('-inf')
    stack, curr = [], root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        if curr.val <= prev:
            return False
        prev = curr.val
        curr = curr.right
    return True
```

---

## 五、评分要点

1. **递归思维** — 能否快速写出递归解法
2. **边界条件** — 空节点、单节点处理
3. **BST性质运用** — 是否理解 BST 的中序有序性
4. **序列化设计** — 能否考虑空节点、分隔符
