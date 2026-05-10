# 最小路径和 (Min Path Sum)

## 1. 概念与定义

最小路径和问题：给定一个 m×n 的非负整数网格，从左上角到右下角，每次只能向右或向下移动一步，求经过的格子数字之和的最小值。

这是网格DP的经典问题，核心思想是：到达每个格子的最小路径和 = 该格子的值 + 从上方或左方到达的最小路径和。

## 2. 状态定义与转移方程

```
dp[i][j] = 从(0,0)到(i,j)的最小路径和
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
dp[0][0] = grid[0][0]
dp[i][0] = dp[i-1][0] + grid[i][0]
dp[0][j] = dp[0][j-1] + grid[0][j]
```

### 空间优化

```
dp[j] = 当前行到第j列的最小路径和
dp[j] = grid[i][j] + min(dp[j], dp[j-1])
```

## 3. 算法实现

### 3.1 标准DP

```python
def minPathSum(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1])
    return dp[m-1][n-1]
```

### 3.2 空间优化

```python
def minPathSum_optimized(grid):
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

### 3.3 原地修改

```python
def minPathSum_inplace(grid):
    m, n = len(grid), len(grid[0])
    for i in range(1, m):
        grid[i][0] += grid[i-1][0]
    for j in range(1, n):
        grid[0][j] += grid[0][j-1]
    for i in range(1, m):
        for j in range(1, n):
            grid[i][j] += min(grid[i-1][j], grid[i][j-1])
    return grid[m-1][n-1]
```

### 3.4 带障碍物的最小路径和

```python
def minPathSumWithObstacles(grid):
    m, n = len(grid), len(grid[0])
    if grid[0][0] < 0 or grid[m-1][n-1] < 0:
        return -1
    INF = float('inf')
    dp = [INF] * n
    dp[0] = 0
    for i in range(m):
        if grid[i][0] < 0:
            dp[0] = INF
        else:
            dp[0] += grid[i][0]
        for j in range(1, n):
            if grid[i][j] < 0:
                dp[j] = INF
            else:
                dp[j] = grid[i][j] + min(dp[j], dp[j-1])
        # 如果全部不可达
        if dp[n-1] == INF and all(x == INF for x in dp):
            return -1
    return dp[n-1] if dp[n-1] != INF else -1
```

### 3.5 路径还原

```python
def minPathSum_with_path(grid):
    m, n = len(grid), len(grid[0])
    dp = [[0] * n for _ in range(m)]
    parent = [[None] * n for _ in range(m)]

    dp[0][0] = grid[0][0]
    for i in range(1, m):
        dp[i][0] = dp[i-1][0] + grid[i][0]
        parent[i][0] = 'D'
    for j in range(1, n):
        dp[0][j] = dp[0][j-1] + grid[0][j]
        parent[0][j] = 'R'

    for i in range(1, m):
        for j in range(1, n):
            if dp[i-1][j] < dp[i][j-1]:
                dp[i][j] = grid[i][j] + dp[i-1][j]
                parent[i][j] = 'D'
            else:
                dp[i][j] = grid[i][j] + dp[i][j-1]
                parent[i][j] = 'R'

    path = []
    i, j = m-1, n-1
    while parent[i][j] is not None:
        path.append(parent[i][j])
        if parent[i][j] == 'D':
            i -= 1
        else:
            j -= 1
    path.reverse()
    return dp[m-1][n-1], ''.join(path)
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 二维DP | O(mn) | O(mn) |
| 一维DP | O(mn) | O(n) |
| 原地修改 | O(mn) | O(1) |

## 5. 典型例题

### 例题1：三角形最小路径和（LeetCode 120）

```python
def minimumTotal(triangle):
    """从顶到底的最小路径和"""
    n = len(triangle)
    dp = triangle[-1][:]  # 从最后一行开始
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
    return dp[0]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **边界处理**：第一行和第一列只有单一来源
2. **负数处理**：如果允许负数，dp初始化和比较需要调整
3. **溢出问题**：大数求和可能溢出

### 6.2 扩展变体

- **只能走某些格子**：跳过障碍物
- **可以走对角线**：三方取min
- **最大路径和**：把min换成max
- **路径计数同时求最小和**：多维DP
