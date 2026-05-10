# 不同路径 (Unique Paths)

## 1. 概念与定义

不同路径问题是网格DP的经典入门问题。在一个 m×n 的网格中，机器人从左上角 (0,0) 出发，每次只能向右或向下移动一步，问到达右下角 (m-1, n-1) 有多少条不同路径。

这类问题的变体包括：
- 带障碍物的路径
- 最小路径和
- 路径上的最大/最小值
- 路径还原

## 2. 状态定义与转移方程

### 2.1 基本路径计数

```
dp[i][j] = 从(0,0)到(i,j)的不同路径数
dp[i][j] = dp[i-1][j] + dp[i][j-1]
dp[0][0] = 1, dp[i][0] = 1, dp[0][j] = 1
```

### 2.2 带障碍物

```
dp[i][j] = 0 if obstacle[i][j] == 1
dp[i][j] = dp[i-1][j] + dp[i][j-1] otherwise
```

### 2.3 数学公式

```
总步数 = (m-1) + (n-1) = m + n - 2
向下步数 = m - 1
答案 = C(m+n-2, m-1)
```

## 3. 算法实现

### 3.1 标准DP

```python
def uniquePaths(m, n):
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]
```

### 3.2 空间优化

```python
def uniquePaths_optimized(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[n-1]
```

### 3.3 数学方法

```python
def uniquePaths_math(m, n):
    from math import comb
    return comb(m + n - 2, m - 1)
```

### 3.4 带障碍物（LeetCode 63）

```python
def uniquePathsWithObstacles(grid):
    m, n = len(grid), len(grid[0])
    if grid[0][0] == 1:
        return 0
    dp = [0] * n
    dp[0] = 1
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 1:
                dp[j] = 0
            elif j > 0:
                dp[j] += dp[j-1]
    return dp[n-1]
```

### 3.5 路径还原

```python
def uniquePaths_with_path(m, n):
    dp = [[0] * n for _ in range(m)]
    parent = [[None] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    # 路径还原需要额外记录来源
    path = []
    i, j = m-1, n-1
    while i > 0 or j > 0:
        if i == 0:
            path.append('R'); j -= 1
        elif j == 0:
            path.append('D'); i -= 1
        else:
            path.append('D'); i -= 1  # 简化：优先向下
    path.reverse()
    return dp[m-1][n-1], ''.join(path)
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 二维DP | O(mn) | O(mn) |
| 一维DP | O(mn) | O(n) |
| 数学公式 | O(min(m,n)) | O(1) |

## 5. 典型例题

### 例题1：最小路径和（LeetCode 64）

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

### 例题2：不同路径III（LeetCode 980）

```python
def uniquePathsIII(grid):
    """必须遍历所有空格恰好一次"""
    m, n = len(grid), len(grid[0])
    empty = 0
    start = None
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 0:
                empty += 1
            elif grid[i][j] == 1:
                start = (i, j)

    result = [0]
    def dfs(i, j, remain):
        if grid[i][j] == 2:
            if remain == 0:
                result[0] += 1
            return
        temp = grid[i][j]
        grid[i][j] = -1
        for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] != -1:
                dfs(ni, nj, remain - 1)
        grid[i][j] = temp

    dfs(start[0], start[1], empty + 1)
    return result[0]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **大数溢出**：路径数可能很大，需要取模
2. **障碍物在起点/终点**：直接返回0
3. **空间优化后无法还原路径**

### 6.2 变体扩展

- **只能走某些格子**：格子有颜色/类型限制
- **对角线移动**：可以四个方向
- **三维路径**：从 (0,0,0) 到 (a,b,c)
