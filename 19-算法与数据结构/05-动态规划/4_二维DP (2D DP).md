# 二维DP (2D DP)

## 1. 概念与定义

二维DP是指状态需要两个维度来描述的动态规划问题。通常用 `dp[i][j]` 表示。二维DP广泛应用于网格路径问题、两个序列的匹配问题等。

二维DP的常见场景：
- **网格DP**：在二维网格上的路径问题
- **双串DP**：涉及两个字符串/序列的DP，如LCS、编辑距离
- **区间DP**：`dp[i][j]` 表示区间 [i, j] 上的最优值

## 2. 状态定义与转移方程

### 2.1 网格路径计数
```
dp[i][j] = dp[i-1][j] + dp[i][j-1]
dp[0][0] = 1, dp[i][0] = 1, dp[0][j] = 1
```

### 2.2 最小路径和
```
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
dp[0][0] = grid[0][0]
```

### 2.3 带障碍物的路径
```
dp[i][j] = 0 if obstacle[i][j] else dp[i-1][j] + dp[i][j-1]
```

### 2.4 最大正方形
```
dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
```

## 3. 算法实现

### 3.1 不同路径（LeetCode 62）

```python
def uniquePaths(m, n):
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]

# 空间优化
def uniquePaths_optimized(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j - 1]
    return dp[n - 1]

# 数学方法
def uniquePaths_math(m, n):
    from math import comb
    return comb(m + n - 2, m - 1)
```

### 3.2 最小路径和（LeetCode 64）

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i - 1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j - 1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i - 1][j], dp[i][j - 1])
    return dp[m - 1][n - 1]
```

### 3.3 不同路径 II（LeetCode 63）

```python
def uniquePathsWithObstacles(grid):
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1:
        return 0
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = 1
    for i in range(1, m):
        dp[i][0] = 0 if grid[i][0] == 1 else dp[i - 1][0]
    for j in range(1, n):
        dp[0][j] = 0 if grid[0][j] == 1 else dp[0][j - 1]
    for i in range(1, m):
        for j in range(1, n):
            if grid[i][j] == 1:
                dp[i][j] = 0
            else:
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
    return dp[m - 1][n - 1]
```

### 3.4 最大正方形（LeetCode 221）

```python
def maximalSquare(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    dp = [[0] * n for _ in range(m)]
    max_side = 0
    for i in range(m):
        for j in range(n):
            if matrix[i][j] == '1':
                if i == 0 or j == 0:
                    dp[i][j] = 1
                else:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                max_side = max(max_side, dp[i][j])
    return max_side * max_side
```

### 3.5 地下城游戏（LeetCode 174）

```python
def calculateMinimumHP(dungeon):
    """从右下角反向DP"""
    m, n = len(dungeon), len(dungeon[0])
    dp = [[float('inf')] * (n + 1) for _ in range(m + 1)]
    dp[m][n - 1] = dp[m - 1][n] = 1
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            need = min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j]
            dp[i][j] = max(1, need)
    return dp[0][0]
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 | 优化后空间 |
|------|-----------|-----------|-----------|
| 不同路径 | O(mn) | O(mn) | O(n) |
| 最小路径和 | O(mn) | O(mn) | O(n) |
| 最大正方形 | O(mn) | O(mn) | O(n) |
| 地下城游戏 | O(mn) | O(mn) | O(n) |

## 5. 典型例题

### 例题1：出界的路径数（LeetCode 576）

```python
def findPaths(m, n, maxMove, startRow, startColumn):
    MOD = 10**9 + 7
    dp = [[0] * n for _ in range(m)]
    dp[startRow][startColumn] = 1
    result = 0
    dirs = [(0,1),(0,-1),(1,0),(-1,0)]
    for _ in range(maxMove):
        new_dp = [[0] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                if dp[i][j] > 0:
                    for di, dj in dirs:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            new_dp[ni][nj] = (new_dp[ni][nj] + dp[i][j]) % MOD
                        else:
                            result = (result + dp[i][j]) % MOD
        dp = new_dp
    return result
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **边界条件遗漏**：第一行和第一列需要单独初始化
2. **越界访问**：转移时需要检查 `i-1 >= 0` 和 `j-1 >= 0`
3. **障碍物处理**：障碍物位置的 dp 值应该为 0

### 6.2 空间优化技巧

```python
# 原始二维
dp[i][j] = dp[i-1][j] + dp[i][j-1]

# 优化为一维
dp[j] = dp[j] + dp[j-1]
# dp[j]保存的是上一行的值，dp[j-1]是当前行已更新的值
```

### 6.3 遍历顺序

- 标准网格DP：从左到右、从上到下
- 反向DP：从右下到左上（如地下城游戏）
- 需要根据转移依赖关系确定顺序
