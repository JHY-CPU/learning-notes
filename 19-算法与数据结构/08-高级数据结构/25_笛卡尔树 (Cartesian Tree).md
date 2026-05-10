# 笛卡尔树 (Cartesian Tree)

## 1. 概述

笛卡尔树（Cartesian Tree）是一种同时满足**二叉搜索树性质**和**堆性质**的二叉树。给定一个序列，笛卡尔树的构造是唯一确定的。

核心性质：
- **中序遍历**：等于原序列
- **堆性质**：通常为最小堆（每个节点的值 <= 子节点的值）

## 2. 定义

给定序列 a[0..n-1]，笛卡尔树满足：
1. 树的中序遍历恰好是 a[0], a[1], ..., a[n-1]
2. 对于每个节点 i，a[i] 小于其所有后代节点的值（最小堆性质）

### 2.1 示例

序列 [3, 2, 6, 1, 9, 5] 的笛卡尔树：

```
        1
       / \
      2   5
     /   / \
    3   6   9
```

验证：
- 中序遍历：3, 2, 6, 1, 9, 5 ✓
- 最小堆性质：每个节点的值 < 子节点 ✓

## 3. 线性构建算法

### 3.1 单调栈方法

使用单调递增栈，O(n) 构建笛卡尔树：
1. 从左到右遍历序列
2. 维护一个栈，栈内元素的值单调递增
3. 对于新元素 a[i]，弹出栈中所有 > a[i] 的元素
4. a[i] 的左子节点是最后一个被弹出的元素
5. 栈顶元素的右子节点设为 a[i]

### 3.2 代码实现

```python
def build_cartesian_tree(arr):
    """
    O(n) 构建笛卡尔树（最小堆性质）
    返回根节点
    """
    n = len(arr)
    if n == 0:
        return None

    nodes = [CartesianNode(i, arr[i]) for i in range(n)]
    stack = []  # 单调递增栈（存下标）

    root = None

    for i in range(n):
        last = None

        # 弹出所有 > arr[i] 的元素
        while stack and arr[stack[-1]] > arr[i]:
            last = stack.pop()

        # last 是最后一个被弹出的，成为 i 的左子节点
        if last is not None:
            nodes[i].left = nodes[last]

        # 栈顶元素的右子节点为 i
        if stack:
            nodes[stack[-1]].right = nodes[i]
        else:
            root = nodes[i]  # i 成为新的根

        stack.append(i)

    return root
```

### 3.3 C++ 实现

```cpp
struct CartesianNode {
    int idx;    // 原数组下标
    int val;    // 值
    int left;   // 左子节点下标
    int right;  // 右子节点下标

    CartesianNode() : idx(-1), val(0), left(-1), right(-1) {}
};

int buildCartesianTree(int arr[], int n, CartesianNode nodes[]) {
    stack<int> stk;
    int root = -1;

    for (int i = 0; i < n; i++) {
        nodes[i].idx = i;
        nodes[i].val = arr[i];
        nodes[i].left = nodes[i].right = -1;

        int last = -1;
        while (!stk.empty() && arr[stk.top()] > arr[i]) {
            last = stk.top();
            stk.pop();
        }

        if (last != -1)
            nodes[i].left = last;

        if (!stk.empty())
            nodes[stk.top()].right = i;
        else
            root = i;

        stk.push(i);
    }

    return root;
}
```

## 4. 节点定义

```python
class CartesianNode:
    """笛卡尔树节点"""
    def __init__(self, idx, val):
        self.idx = idx    # 原数组下标
        self.val = val    # 值
        self.left = None
        self.right = None

    def __repr__(self):
        return f"CartesianNode(idx={self.idx}, val={self.val})"
```

## 5. RMQ应用（核心）

### 5.1 原理

笛卡尔树天然支持**区间最小值查询（RMQ）**：
- 序列 a[l..r] 的最小值，就是笛卡尔树中 l 和 r 的最近公共祖先（LCA）的值

因为：
- 中序遍历 = 原序列，所以下标 l 到 r 之间的节点在中序遍历中连续
- 堆性质保证最小值是这些节点在树中的LCA

### 5.2 LCA查询

```python
def lca(root, u, v):
    """在笛卡尔树上求LCA"""
    if root is None:
        return None

    if root.idx == u or root.idx == v:
        return root

    left_lca = lca(root.left, u, v)
    right_lca = lca(root.right, u, v)

    if left_lca and right_lca:
        return root  # u和v分别在左右子树

    return left_lca if left_lca else right_lca

def rmq(root, l, r):
    """通过笛卡尔树查询区间最小值"""
    lca_node = lca(root, l, r)
    return lca_node.val if lca_node else None
```

### 5.3 O(1) LCA

预处理后用RMQ可以在O(1)时间内求LCA，从而：
- 构建笛卡尔树：O(n)
- 预处理LCA（欧拉序+RMQ）：O(n)
- 单次RMQ查询：O(1)

这给出了一种 O(n) 预处理、O(1) 查询的RMQ解法。

## 6. 完整示例

```python
if __name__ == "__main__":
    arr = [3, 2, 6, 1, 9, 5, 4, 7, 8]
    root = build_cartesian_tree(arr)

    print("序列:", arr)
    print("笛卡尔树根:", root)

    # 验证中序遍历
    def inorder(node):
        if node is None:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)

    print("中序遍历:", inorder(root))  # 应等于原序列

    # RMQ查询
    print(f"RMQ(2, 5) = min(arr[2..5]) = {rmq(root, 2, 5)}")  # min(6,1,9,5) = 1
    print(f"RMQ(0, 3) = min(arr[0..3]) = {rmq(root, 0, 3)}")  # min(3,2,6,1) = 1
```

## 7. 最大笛卡尔树

如果使用最大堆性质（每个节点 >= 子节点），构建方法类似，只需改变比较方向。

```python
def build_max_cartesian_tree(arr):
    """构建最大堆笛卡尔树"""
    n = len(arr)
    nodes = [CartesianNode(i, arr[i]) for i in range(n)]
    stack = []

    for i in range(n):
        last = None
        while stack and arr[stack[-1]] < arr[i]:  # 改为 <
            last = stack.pop()

        if last is not None:
            nodes[i].left = nodes[last]

        if stack:
            nodes[stack[-1]].right = nodes[i]

        stack.append(i)

    return nodes[stack[0]] if stack else None
```

## 8. 复杂度分析

| 操作 | 时间复杂度 |
|------|-----------|
| 构建 | O(n) |
| RMQ查询 | O(log n) 或 O(1)（配合LCA） |
| 中序遍历 | O(n) |
| 空间 | O(n) |

## 9. Treap 与笛卡尔树的关系

- Treap 可以看作随机化的笛卡尔树
- 如果 priority 随机分配，Treap 是序列按 key 排序后对应的笛卡尔树
- 笛卡尔树的 key 对应序列值，priority 对应下标

## 10. 应用场景

1. 区间最小值查询（RMQ）
2. 作为其他算法的中间结构
3. Treap的理论基础
4. 某些动态规划优化

## 11. 总结

笛卡尔树是一种将序列的"搜索"和"堆"两种性质完美结合的数据结构：
- O(n) 线性构建
- 中序遍历 = 原序列
- LCA = RMQ
- 与Treap有深刻联系
