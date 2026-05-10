# 树形DP (Tree DP)

## 1. 概念与定义

树形DP是在树结构上进行的动态规划。利用树的递归性质，通过后序遍历（先处理子节点，再处理父节点），将子树的最优解合并得到整棵树的最优解。

树形DP的特点：
- 状态通常与**树的节点**相关
- 转移通过**父子关系**进行
- 遍历顺序为**后序遍历**（DFS）
- 时间复杂度通常为 O(n) 或 O(n²)

常见应用：
- 树的最大独立集
- 树的直径
- 树的最小支配集
- 树上背包问题

## 2. 状态定义与转移方程

### 2.1 树的最大独立集

```
dp[u][0] = 以u为根的子树，不选u时的最大独立集大小
dp[u][1] = 以u为根的子树，选u时的最大独立集大小

转移：
  dp[u][0] = sum(max(dp[v][0], dp[v][1]))  for v in children(u)
  dp[u][1] = 1 + sum(dp[v][0])             for v in children(u)
```

### 2.2 树的直径

```
dp[u] = 以u为根的子树中，从u出发的最长路径
直径 = max(dp[u] + dp[v] + w(u,v)) 对所有节点u和子节点v

转移：dp[u] = max(dp[v] + w(u,v)) for v in children(u)
```

### 2.3 树的最小支配集

```
dp[u][0] = u被支配，u的父节点不需要被u支配
dp[u][1] = u被支配，u的父节点需要被u支配
dp[u][2] = u未被支配（需要子节点支配它）
```

### 2.4 打家劫舍III（树形）

```
dp[u][0] = 不偷u节点的最大金额
dp[u][1] = 偷u节点的最大金额

dp[u][0] = sum(max(dp[v][0], dp[v][1])) for v in children(u)
dp[u][1] = val[u] + sum(dp[v][0])       for v in children(u)
```

## 3. 算法实现

### 3.1 打家劫舍III（LeetCode 337）

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def rob(root):
    def dfs(node):
        if not node:
            return [0, 0]  # [不偷, 偷]
        left = dfs(node.left)
        right = dfs(node.right)

        # 不偷当前节点：子节点可偷可不偷
        not_rob = max(left) + max(right)
        # 偷当前节点：子节点不能偷
        do_rob = node.val + left[0] + right[0]

        return [not_rob, do_rob]

    return max(dfs(root))
```

### 3.2 树的直径（LeetCode 543）

```python
def diameterOfBinaryTree(root):
    result = 0

    def dfs(node):
        nonlocal result
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        result = max(result, left + right)  # 更新直径
        return max(left, right) + 1  # 返回从当前节点出发的最长路径

    dfs(root)
    return result
```

### 3.3 树的最大独立集

```python
def maxIndependentSet(root):
    def dfs(node):
        if not node:
            return [0, 0]  # [不选当前节点, 选当前节点]
        left = dfs(node.left)
        right = dfs(node.right)
        # 不选当前节点：子节点可选可不选
        not_select = max(left) + max(right)
        # 选当前节点：子节点不能选
        select = 1 + left[0] + right[0]
        return [not_select, select]

    return max(dfs(root))
```

### 3.4 树上背包（子树合并）

```python
def treeKnapsack(root, cost, value, W):
    """
    树上背包：每个节点是一个物品，选父节点必须选子节点
    dp[u][j] = 以u为根的子树、容量为j时的最大价值
    """
    def dfs(u):
        dp = [[0] * (W + 1) for _ in range(2)]
        # dp[0]：不选u，dp[1]：选u
        dp[1][cost[u]] = value[u] if cost[u] <= W else 0

        for v in children[u]:
            child_dp = dfs(v)
            new_dp = [[0] * (W + 1) for _ in range(2)]
            for j in range(W + 1):
                for k in range(j + 1):
                    new_dp[0][j] = max(new_dp[0][j], dp[0][k] + max(child_dp[0][j-k], child_dp[1][j-k]))
                    if k >= cost[u]:
                        new_dp[1][j] = max(new_dp[1][j], dp[1][k] + child_dp[1][j-k])
            dp = new_dp
        return dp

    return max(dfs(root))
```

### 3.5 C++ 实现

```cpp
// 树的直径
int ans = 0;
int dfs(TreeNode* node) {
    if (!node) return 0;
    int l = dfs(node->left);
    int r = dfs(node->right);
    ans = max(ans, l + r);
    return max(l, r) + 1;
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 打家劫舍III | O(n) | O(h)递归栈 |
| 树的直径 | O(n) | O(h) |
| 最大独立集 | O(n) | O(h) |
| 树上背包 | O(nW²) | O(nW) |

## 5. 典型例题

### 例题1：二叉树的最长交错路径（LeetCode 1372）

```python
def longestZigZag(root):
    result = 0

    def dfs(node):
        nonlocal result
        if not node:
            return [-1, -1]  # [左子树方向, 右子树方向]
        left = dfs(node.left)
        right = dfs(node.right)
        result = max(result, left[1] + 1, right[0] + 1)
        return [left[1] + 1, right[0] + 1]

    dfs(root)
    return result
```

### 例题2：监控二叉树（LeetCode 968）

```python
def minCameraCover(root):
    """
    状态：0=未覆盖，1=有摄像头，2=已覆盖
    """
    result = 0

    def dfs(node):
        nonlocal result
        if not node:
            return 2  # 空节点视为已覆盖
        left = dfs(node.left)
        right = dfs(node.right)

        if left == 0 or right == 0:
            result += 1
            return 1  # 需要安装摄像头

        if left == 1 or right == 1:
            return 2  # 已被子节点覆盖

        return 0  # 未被覆盖

    if dfs(root) == 0:
        result += 1
    return result
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **遍历顺序**：必须后序遍历，先处理子节点
2. **空节点处理**：递归终止条件需要正确处理空节点
3. **全局变量**：直径、最值等需要在遍历过程中更新
4. **有根树vs无根树**：无根树需要任意选一个根，用邻接表存储

### 6.2 优化方向

1. **树上启发式合并**（DSU on Tree）：O(nlogn) 解决子树查询
2. **树链剖分**：将树分解为若干条链，用线段树维护
3. **换根DP**：枚举每个节点作为根，通过两次DFS求解

### 6.3 树形DP模板

```python
def tree_dp(root):
    def dfs(u, parent):
        # 初始化当前节点的状态
        # 遍历所有子节点
        for v in children[u]:
            if v == parent:
                continue
            dfs(v, u)
            # 用子节点的状态更新当前节点
        # 计算最终答案

    dfs(root, -1)
    return 答案
```
