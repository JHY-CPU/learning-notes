# 滚动数组 (Rolling Array)

## 1. 概念与定义

滚动数组是动态规划中**空间优化**的核心技巧。当DP状态转移只依赖于前几行/列时，可以用滚动数组将空间从 O(n) 降到 O(1) 或 O(k)（k为依赖的行数）。

核心思想：**重复利用数组空间**，通过取模或交替使用两个数组来保存必要的状态。

## 2. 常见滚动方式

### 2.1 双变量滚动（O(1) 空间）

```
适用于：dp[i] 只依赖 dp[i-1] 和 dp[i-2]
a, b = b, a + b  （斐波那契）
```

### 2.2 一维数组滚动（O(n) 空间）

```
适用于：dp[i][j] 只依赖 dp[i-1][j] 和 dp[i][j-1]
dp[j] = dp[j] + dp[j-1]  （不同路径）
```

### 2.3 取模滚动

```
适用于：dp[i] 依赖 dp[i-k] 到 dp[i-1]
dp[i % k][j] = f(dp[(i-1) % k][j])
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

# O(1) 空间 — 滚动变量
def fib_rolling(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```

### 3.2 不同路径

```python
# O(mn) 空间
def uniquePaths_2d(m, n):
    dp = [[0] * n for _ in range(m)]
    for i in range(m): dp[i][0] = 1
    for j in range(n): dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# O(n) 空间 — 滚动数组
def uniquePaths_rolling(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]  # dp[j]是上一行，dp[j-1]是当前行
    return dp[n-1]
```

### 3.3 0-1背包

```python
# O(nW) 空间
def knapsack_2d(w, v, W):
    n = len(w)
    dp = [[0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(W + 1):
            dp[i][j] = dp[i-1][j]
            if j >= w[i-1]:
                dp[i][j] = max(dp[i][j], dp[i-1][j-w[i-1]] + v[i-1])
    return dp[n][W]

# O(W) 空间 — 滚动数组 + 逆序
def knapsack_rolling(w, v, W):
    dp = [0] * (W + 1)
    for i in range(len(w)):
        for j in range(W, w[i] - 1, -1):  # 逆序！
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    return dp[W]
```

### 3.4 编辑距离

```python
# O(mn) 空间
def editDistance_2d(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# O(n) 空间 — 滚动数组
def editDistance_rolling(s1, s2):
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

### 3.5 依赖多行的滚动

```python
# 如果 dp[i] 依赖 dp[i-1], dp[i-2], dp[i-3]
# 可以用 (i % 3) 来滚动
def dp_three_rows(n):
    dp = [[0] * n for _ in range(3)]
    for i in range(n):
        for j in range(n):
            dp[i % 3][j] = dp[(i-1) % 3][j] + dp[(i-2) % 3][j] + dp[(i-3) % 3][j]
```

## 4. 复杂度分析

| 问题 | 原始空间 | 滚动后空间 |
|------|---------|-----------|
| 斐波那契 | O(n) | O(1) |
| 不同路径 | O(mn) | O(n) |
| 0-1背包 | O(nW) | O(W) |
| 编辑距离 | O(mn) | O(n) |
| LCS | O(mn) | O(min(m,n)) |

## 5. 常见陷阱

### 5.1 陷阱

1. **0-1背包必须逆序**：正序会导致物品被重复选取
2. **完全背包必须正序**：逆序会变成0-1背包
3. **保存中间值**：滚动数组无法回溯路径
4. **多行依赖**：需要正确取模

### 5.2 何时不能用滚动数组

- 需要回溯/还原路径时
- 转移依赖很多非相邻状态时
- 需要随机访问所有状态时

### 5.3 检查方法

在写DP时，画出二维DP表，标出每个位置依赖的上一层位置。如果只有少数几个位置被依赖，就可以滚动。

```
二维dp表：
  dp[i-1][j-1]  dp[i-1][j]
  dp[i][j-1]    dp[i][j]  ← 计算这里

只依赖上面和左边 → 可以一维滚动
依赖上面两行 → 需要保存两行或取模
```
