# DP解题模板 (DP Templates)

## 1. DP解题五步法

```
步骤1：定义状态 — dp[i] 或 dp[i][j] 代表什么含义
步骤2：写转移方程 — 当前状态如何由子状态推导
步骤3：确定边界 — 最小子问题的解
步骤4：确定遍历顺序 — 保证转移时子问题已求解
步骤5：返回答案 — dp[n] 或 max(dp) 等
```

## 2. 一维DP模板

### 2.1 基本模板

```python
def solve_1d(nums):
    n = len(nums)
    if n == 0:
        return 0

    # 1. 定义状态
    dp = [0] * n

    # 2. 边界条件
    dp[0] = nums[0]  # 根据题意设置

    # 3. 递推
    for i in range(1, n):
        dp[i] = 状态转移  # 如 max(dp[i-1], dp[i-2] + nums[i])

    # 4. 返回答案
    return dp[n-1]  # 或 max(dp)
```

### 2.2 空间优化模板

```python
def solve_1d_optimized(nums):
    n = len(nums)
    if n == 0: return 0
    if n == 1: return nums[0]

    prev2, prev1 = nums[0], max(nums[0], nums[1])
    for i in range(2, n):
        prev2, prev1 = prev1, max(prev1, prev2 + nums[i])
    return prev1
```

## 3. 二维DP模板

### 3.1 网格DP

```python
def solve_grid(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]

    # 初始化
    dp[0][0] = grid[0][0]
    for i in range(1, m): dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n): dp[0][j] = dp[0][j-1] + grid[0][j]

    # 递推
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])

    return dp[m-1][n-1]
```

### 3.2 空间优化

```python
def solve_grid_optimized(grid):
    m, n = len(grid), len(grid[0])
    dp = [0] * n
    dp[0] = grid[0][0]
    for j in range(1, n): dp[j] = dp[j-1] + grid[0][j]

    for i in range(1, m):
        dp[0] += grid[i][0]
        for j in range(1, n):
            dp[j] = grid[i][j] + min(dp[j], dp[j-1])

    return dp[n-1]
```

## 4. 背包DP模板

### 4.1 0-1背包

```python
def knapsack_01(w, v, W):
    dp = [0] * (W + 1)
    for i in range(len(w)):
        for j in range(W, w[i] - 1, -1):  # 逆序
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    return dp[W]
```

### 4.2 完全背包

```python
def knapsack_complete(w, v, W):
    dp = [0] * (W + 1)
    for i in range(len(w)):
        for j in range(w[i], W + 1):  # 正序
            dp[j] = max(dp[j], dp[j - w[i]] + v[i])
    return dp[W]
```

### 4.3 背包方案数

```python
def knapsack_count(w, W):
    dp = [0] * (W + 1)
    dp[0] = 1
    for i in range(len(w)):
        for j in range(W, w[i] - 1, -1):  # 0-1背包逆序
            dp[j] += dp[j - w[i]]
    return dp[W]
```

## 5. 区间DP模板

```python
def solve_interval(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]

    # 初始化
    for i in range(n):
        dp[i][i] = 初始值

    # 按区间长度递推
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = 极值
            for k in range(i, j):
                dp[i][j] = 最优(dp[i][j], dp[i][k] + dp[k+1][j] + cost(i,j,k))

    return dp[0][n-1]
```

## 6. 记忆化搜索模板

```python
from functools import lru_cache

def solve_memo(params):
    @lru_cache(maxsize=None)
    def dp(状态):
        if 边界条件:
            return 基础值
        result = 初始值
        for 选择 in 所有选择:
            result = 最优(result, dp(新状态))
        return result

    return dp(初始状态)
```

## 7. 状态压缩DP模板

```python
def solve_bitmask(n):
    dp = [[float('inf')] * n for _ in range(1 << n)]

    # 初始化
    for i in range(n):
        dp[1 << i][i] = 初始值

    for mask in range(1 << n):
        for u in range(n):
            if not (mask >> u) & 1 or dp[mask][u] == float('inf'):
                continue
            for v in range(n):
                if (mask >> v) & 1:
                    continue
                new_mask = mask | (1 << v)
                dp[new_mask][v] = min(dp[new_mask][v], dp[mask][u] + cost(u, v))

    return min(dp[(1 << n) - 1])
```

## 8. 数位DP模板

```python
def solve_digit(n):
    digits = list(map(int, str(n)))
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(pos, state, limit):
        if pos == len(digits):
            return 边界判断(state)
        upper = digits[pos] if limit else 9
        result = 0
        for d in range(upper + 1):
            if 合法(state, d):
                result += dfs(pos + 1, 新状态(state, d), limit and d == upper)
        return result

    return dfs(0, 初始状态, True)
```

## 9. 股票买卖模板

```python
def stock_dp(prices, k):
    n = len(prices)
    if k >= n // 2:  # 等价于无限次
        return sum(max(0, prices[i] - prices[i-1]) for i in range(1, n))

    dp = [[0, -prices[0]] for _ in range(k + 1)]
    for i in range(1, n):
        for j in range(k, 0, -1):  # 逆序
            dp[j][0] = max(dp[j][0], dp[j][1] + prices[i])
            dp[j][1] = max(dp[j][1], dp[j-1][0] - prices[i])
    return dp[k][0]
```
