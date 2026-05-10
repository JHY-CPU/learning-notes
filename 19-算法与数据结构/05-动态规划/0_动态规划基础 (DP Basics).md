# 动态规划基础 (DP Basics)

## 1. 概念与定义

动态规划（Dynamic Programming，简称DP）是一种通过把原问题分解为相对简单的子问题的方式求解复杂问题的方法。动态规划常常适用于有**重叠子问题**和**最优子结构性**质的问题，动态规划方法所耗时间往往远少于朴素的穷举算法。

动态规划的核心思想：
- **最优子结构**：问题的最优解包含子问题的最优解。即可以通过组合子问题的最优解来构造原问题的最优解。
- **重叠子问题**：在递归求解过程中，子问题被反复计算多次。动态规划通过保存已计算的结果避免重复计算。
- **无后效性**：某阶段的状态一旦确定，就不再受后续决策的影响。

动态规划与分治法的区别在于，分治法分解出的子问题是互相独立的，而动态规划分解出的子问题是互相重叠的。

## 2. 状态定义与转移方程

### 2.1 状态定义

状态定义是DP问题的核心。一个好的状态定义应该满足：
1. **完备性**：状态能够完整描述当前子问题的所有信息
2. **无后效性**：确定状态后，不受后续决策影响
3. **最优子结构性**：当前状态的最优解可以由子状态的最优解推导

### 2.2 状态转移方程

状态转移方程描述了状态之间的递推关系。一般形式为：

```
dp[i] = f(dp[j])  其中 j < i（或 j 是 i 的子状态）
```

以斐波那契数列为例：
```
dp[i] = dp[i-1] + dp[i-2]
初始条件：dp[0] = 0, dp[1] = 1
```

以0-1背包为例：
```
dp[i][j] = max(dp[i-1][j], dp[i-1][j-w[i]] + v[i])
dp[i][j] 表示前 i 个物品、容量为 j 时的最大价值
```

### 2.3 推导转移方程的步骤

1. **明确状态含义**：确定 `dp[i]` 或 `dp[i][j]` 代表什么
2. **寻找子问题**：当前问题可以分解为哪些更小的子问题
3. **考虑所有选择**：当前状态可以由哪些子状态转移而来，取最优
4. **确定边界条件**：最小的子问题的解是什么

## 3. 算法实现

### 3.1 自底向上（表格法 / Tabulation）

```python
def fibonacci(n):
    """自底向上的动态规划"""
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

# 空间优化版本
def fibonacci_optimized(n):
    if n <= 1:
        return n
    prev, curr = 0, 1
    for i in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```

### 3.2 自顶向下（记忆化搜索 / Memoization）

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    """自顶向下的记忆化搜索"""
    if n <= 1:
        return n
    return fibonacci_memo(n - 1) + fibonacci_memo(n - 2)
```

### 3.3 0-1背包问题示例

```python
def knapsack_01(weights, values, capacity):
    """
    0-1背包问题
    weights: 每个物品的重量
    values: 每个物品的价值
    capacity: 背包容量
    """
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(capacity + 1):
            dp[i][j] = dp[i - 1][j]
            if j >= weights[i - 1]:
                dp[i][j] = max(dp[i][j], dp[i - 1][j - weights[i - 1]] + values[i - 1])

    return dp[n][capacity]

# 空间优化版本
def knapsack_01_optimized(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 斐波那契（DP） | O(n) | O(1)优化后 |
| 0-1背包 | O(nW) | O(W)优化后 |
| 最长公共子序列 | O(mn) | O(min(m,n))优化后 |
| 矩阵链乘法 | O(n³) | O(n²) |

## 5. 典型例题

### 例题1：爬楼梯（LeetCode 70）

**问题**：假设你正在爬楼梯。需要 n 阶你才能到达楼顶。每次你可以爬 1 或 2 个台阶。有多少种不同的方法？

**分析**：到达第n阶的方式 = 从第n-1阶爬1步 + 从第n-2阶爬2步

```python
def climbStairs(n):
    if n <= 2:
        return n
    a, b = 1, 2
    for i in range(3, n + 1):
        a, b = b, a + b
    return b
```

### 例题2：最小花费爬楼梯（LeetCode 746）

```python
def minCostClimbingStairs(cost):
    n = len(cost)
    dp = [0] * (n + 1)
    for i in range(2, n + 1):
        dp[i] = min(dp[i-1] + cost[i-1], dp[i-2] + cost[i-2])
    return dp[n]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **边界条件遗漏**：忘记初始化 dp[0]、dp[1] 等基础状态
2. **遍历顺序错误**：0-1背包中如果正序遍历会导致一个物品被多次选取
3. **状态定义不清晰**：没有明确定义 dp[i] 的含义导致转移方程错误
4. **整数溢出**：大数问题需要取模或使用大整数

### 6.2 优化技巧

1. **空间优化**：如果dp[i]只依赖dp[i-1]，可以用滚动数组
2. **剪枝**：提前终止不可能的情况
3. **预处理**：对输入数据排序或预处理减少计算量

### 6.3 DP解题框架

```
1. 定义状态：明确 dp[i] 或 dp[i][j] 的含义
2. 写出转移方程：考虑所有可能的转移
3. 确定边界：最小的子问题
4. 确定遍历顺序：保证转移时子问题已经求解
5. 空间优化（可选）
6. 返回答案
```

### 6.4 如何判断是否使用DP

- 问题要求**最优值**（最大/最小/最长/最短）
- 问题要求**方案数**
- 问题具有**重叠子问题**（可以用递归画出递归树验证）
- 问题具有**最优子结构**
