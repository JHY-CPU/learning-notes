# 矩阵DP (Matrix DP)

## 1. 概念与定义

矩阵DP是指在二维矩阵上进行的动态规划。与网格路径问题不同，矩阵DP涵盖更广泛的问题类型，包括：
- 矩阵中的最大正方形/矩形
- 矩阵中的最大和子矩阵
- 矩阵搜索路径
- 矩阵中的连通区域

## 2. 状态定义与转移方程

### 2.1 最大正方形

```
dp[i][j] = 以(i,j)为右下角的最大正方形边长
dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
dp[i][j] = 1 if matrix[i][j] == '1' and (i==0 or j==0)
```

### 2.2 最大矩形

```
heights[j] = 以当前行为底、第j列的连续1的高度
对每行用单调栈求最大矩形面积
```

### 2.3 矩阵中的最大和子矩阵

```
预处理每列的前缀和
枚举上下边界，转化为一维最大子数组和问题
时间复杂度：O(n²m) 或 O(nm²)
```

## 3. 算法实现

### 3.1 最大正方形（LeetCode 221）

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

### 3.2 最大矩形（LeetCode 85）

```python
def maximalRectangle(matrix):
    if not matrix or not matrix[0]:
        return 0
    m, n = len(matrix), len(matrix[0])
    heights = [0] * n
    max_area = 0

    for i in range(m):
        # 更新柱状图高度
        for j in range(n):
            heights[j] = heights[j] + 1 if matrix[i][j] == '1' else 0

        # 单调栈求最大矩形
        stack = [-1]
        for j in range(n):
            while len(stack) > 1 and heights[stack[-1]] >= heights[j]:
                h = heights[stack.pop()]
                w = j - stack[-1] - 1
                max_area = max(max_area, h * w)
            stack.append(j)
        while len(stack) > 1:
            h = heights[stack.pop()]
            w = n - stack[-1] - 1
            max_area = max(max_area, h * w)

    return max_area
```

### 3.3 最大和子矩阵

```python
def maxSumSubmatrix(matrix, k):
    """和不超过k的最大子矩阵和"""
    import bisect
    m, n = len(matrix), len(matrix[0])
    result = float('-inf')

    for left in range(n):
        row_sum = [0] * m
        for right in range(left, n):
            for i in range(m):
                row_sum[i] += matrix[i][right]

            # 对row_sum求不超过k的最大子数组和
            prefix = [0]
            curr = 0
            max_sum = float('-inf')
            for s in row_sum:
                curr += s
                target = curr - k
                idx = bisect.bisect_left(prefix, target)
                if idx < len(prefix):
                    max_sum = max(max_sum, curr - prefix[idx])
                bisect.insort(prefix, curr)

            result = max(result, max_sum)

    return result
```

### 3.4 矩阵中的最长递增路径（LeetCode 329）

```python
def longestIncreasingPath(matrix):
    m, n = len(matrix), len(matrix[0])

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(i, j):
        best = 1
        for di, dj in [(0,1),(0,-1),(1,0),(-1,0)]:
            ni, nj = i+di, j+dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))
```

### 3.5 C++ 实现

```cpp
int maximalSquare(vector<vector<char>>& matrix) {
    int m = matrix.size(), n = matrix[0].size();
    vector<vector<int>> dp(m, vector<int>(n, 0));
    int maxSide = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            if (matrix[i][j] == '1') {
                if (i == 0 || j == 0) dp[i][j] = 1;
                else dp[i][j] = min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]}) + 1;
                maxSide = max(maxSide, dp[i][j]);
            }
    return maxSide * maxSide;
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 最大正方形 | O(mn) | O(mn) |
| 最大矩形 | O(mn) | O(n) |
| 最大和子矩阵 | O(n²m) | O(m) |
| 最长递增路径 | O(mn) | O(mn) |

## 5. 典型例题

### 例题1：矩阵中的幸运数（LeetCode 2383）

```python
def minSessions(n, sessions):
    """用位运算枚举所有分组方案"""
    pass
```

### 例题2：矩阵中的幻方（LeetCode 840）

```python
def numMagicSquaresInside(grid):
    def is_magic(r, c):
        s = set()
        for i in range(r, r+3):
            for j in range(c, c+3):
                if grid[i][j] < 1 or grid[i][j] > 9 or grid[i][j] in s:
                    return False
                s.add(grid[i][j])
        # 检查行列对角线和
        target = sum(grid[r][c:c+3])
        for i in range(3):
            if sum(grid[r+i][c:c+3]) != target: return False
            if sum(grid[r+j][c+i] for j in range(3)) != target: return False
        if grid[r][c]+grid[r+1][c+1]+grid[r+2][c+2] != target: return False
        if grid[r][c+2]+grid[r+1][c+1]+grid[r+2][c] != target: return False
        return True

    m, n = len(grid), len(grid[0])
    return sum(is_magic(i, j) for i in range(m-2) for j in range(n-2))
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **空矩阵处理**：先检查矩阵是否为空
2. **单行单列**：特殊边界情况
3. **单调栈清空**：最后需要处理栈中剩余元素
4. **记忆化DFS**：矩阵上的DFS容易超时，需要加缓存

### 6.2 优化技巧

1. **前缀和**：矩阵区域求和
2. **单调栈**：柱状图最大矩形
3. **记忆化搜索**：最长递增路径等
4. **滚动数组**：某些问题可以降维
