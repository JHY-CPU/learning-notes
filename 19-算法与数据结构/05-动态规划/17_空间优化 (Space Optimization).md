# 空间优化 (Space Optimization)

## 1. 概念与定义

空间优化是动态规划中重要的优化技巧。许多DP问题中，`dp[i]` 只依赖于 `dp[i-1]`（或更前面的几个状态），此时不需要保存所有状态，只需保留必要的几个即可。

常见的空间优化方法：
1. **滚动数组**：用一维数组代替二维数组
2. **变量替换**：用几个变量代替一维数组
3. **原地修改**：直接在输入数组上修改

## 2. 状态定义与转移方程

### 2.1 二维到一维

```
原始：dp[i][j] = f(dp[i-1][j], dp[i][j-1], ...)
优化：dp[j] = f(dp[j], dp[j-1], ...)
原理：只依赖上一行和当前行已计算的值
```

### 2.2 一维到常数

```
原始：dp[i] = f(dp[i-1], dp[i-2])
优化：a, b = b, f(a, b)
原理：只依赖最近的两个状态
```

### 2.3 遍历方向

```
0-1背包：逆序（保证每个物品只用一次）
完全背包：正序（允许重复选取）
网格DP：通常正序
```

## 3. 算法实现

### 3.1 斐波那契数列

```python
# O(n) 空间
def fib_dp(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]

# O(1) 空间
def fib_optimized(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### 3.2 不同路径（LeetCode 62）

```python
# O(mn) 空间
def uniquePaths_2d(m, n):
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# O(n) 空间
def uniquePaths_1d(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]  # dp[j]是上一行的值，dp[j-1]是当前行已更新的
    return dp[n-1]
```

### 3.3 0-1背包空间优化

```python
# O(nW) 空间
def knapsack_2d(weights, values, W):
    n = len(weights)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(W + 1):
            dp[i][j] = dp[i-1][j]
            if j >= weights[i-1]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-weights[i-1]] + values[i-1])
    return dp[n][W]

# O(W) 空间 — 必须逆序！
def knapsack_1d(weights, values, W):
    dp = [0] * (W + 1)
    for i in range(len(weights)):
        for j in range(W, weights[i] - 1, -1):  # 逆序
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[W]
```

### 3.4 编辑距离空间优化

```python
# O(mn) 空间
def editDistance_2d(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# O(n) 空间 — 需要保存左上角的值
def editDistance_1d(s1, s2):
    m, n = len(s1), len(s2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j-1], prev)
            prev = temp
    return dp[n]
```

### 3.5 打家劫舍空间优化

```python
def rob(nums):
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]
    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, len(nums)):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
    return prev1
```

## 4. 复杂度分析

| 问题 | 原始空间 | 优化后空间 | 优化方法 |
|------|---------|-----------|---------|
| 斐波那契 | O(n) | O(1) | 两个变量 |
| 不同路径 | O(mn) | O(n) | 一维数组 |
| 0-1背包 | O(nW) | O(W) | 一维+逆序 |
| 编辑距离 | O(mn) | O(n) | 一维+prev |
| 打家劫舍 | O(n) | O(1) | 两个变量 |
| LCS | O(mn) | O(min(m,n)) | 一维数组 |

## 5. 典型例题

### 例题1：最小路径和空间优化（LeetCode 64）

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = grid[0][0]
    for j in range(1, n):
        dp[j] = dp[j-1] + grid[0][j]

    for i in range(1, m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])

    return dp[n-1]
```

### 例题2：最长递增子序列（LeetCode 300）

```python
# 这个本身就是 O(n) 空间，不需要优化
# 但可以用 O(nlogn) 时间 + O(n) 空间的贪心+二分
def lengthOfLIS(nums):
    import bisect
    tails = []
    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **0-1背包正序**：正序会导致物品被重复选取，必须逆序
2. **完全背包逆序**：逆序会变成0-1背包，必须正序
3. **保存中间值**：空间优化后无法回溯路径
4. **遍历覆盖**：一维数组中注意不要覆盖还未使用的值

### 6.2 何时可以空间优化

```
可以优化的条件：
1. dp[i] 只依赖 dp[i-1], dp[i-2], ... 中的少数几个
2. 不需要回溯路径
3. 转移方向允许使用更小的空间

不可以优化的情况：
1. 需要回溯/还原路径
2. 转移依赖很多之前的状态
3. 需要随机访问所有状态
```

### 6.3 优化口诀

```
二维降一维：注意遍历方向
一维降常数：只保留最近需要的几个值
背包问题：0-1逆序，完全正序
路径还原：不能做空间优化
```
