# 三角形最小路径和 (Triangle Min Path)

## 1. 概念与定义

给定一个三角形数组，找到从顶到底的最小路径和。每一步只能移动到下一行的**相邻节点**。即从位置 (i, j) 只能移动到 (i+1, j) 或 (i+1, j+1)。

这个问题有多种解法：
- **自顶向下DP**：从第一行开始递推
- **自底向上DP**：从最后一行开始递推（更简洁，不需要处理边界）
- **记忆化搜索**：自顶向下递归 + 缓存

## 2. 状态定义与转移方程

### 2.1 自顶向下

```
dp[i][j] = 从(0,0)到(i,j)的最小路径和
dp[i][j] = triangle[i][j] + min(dp[i-1][j-1], dp[i-1][j])
需要处理 j=0 和 j=i 的边界
答案：min(dp[n-1][0], ..., dp[n-1][n-1])
```

### 2.2 自底向上（推荐）

```
dp[j] = 从最后一行到位置(i,j)的最小路径和
dp[j] = triangle[i][j] + min(dp[j], dp[j+1])
从最后一行开始，逐行向上
答案：dp[0]
```

### 2.3 记忆化搜索

```
dfs(i, j) = 从(i,j)到底部的最小路径和
dfs(i, j) = triangle[i][j] + min(dfs(i+1, j), dfs(i+1, j+1))
边界：i == n-1 时返回 triangle[i][j]
```

## 3. 算法实现

### 3.1 自底向上DP（推荐）

```python
def minimumTotal(triangle):
    n = len(triangle)
    dp = triangle[-1][:]  # 复制最后一行
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[j] = triangle[i][j] + min(dp[j], dp[j + 1])
    return dp[0]
```

### 3.2 自顶向下DP

```python
def minimumTotal_topDown(triangle):
    n = len(triangle)
    dp = [[0] * (i + 1) for i in range(n)]
    dp[0][0] = triangle[0][0]

    for i in range(1, n):
        dp[i][0] = dp[i-1][0] + triangle[i][0]
        for j in range(1, i):
            dp[i][j] = triangle[i][j] + min(dp[i-1][j-1], dp[i-1][j])
        dp[i][i] = dp[i-1][i-1] + triangle[i][i]

    return min(dp[n-1])
```

### 3.3 记忆化搜索

```python
from functools import lru_cache

def minimumTotal_memo(triangle):
    n = len(triangle)

    @lru_cache(maxsize=None)
    def dfs(i, j):
        if i == n - 1:
            return triangle[i][j]
        return triangle[i][j] + min(dfs(i + 1, j), dfs(i + 1, j + 1))

    return dfs(0, 0)
```

### 3.4 空间优化自顶向下

```python
def minimumTotal_optimized(triangle):
    n = len(triangle)
    dp = [0] * n
    dp[0] = triangle[0][0]

    for i in range(1, n):
        # 逆序更新，避免覆盖
        dp[i] = dp[i-1] + triangle[i][i]
        for j in range(i - 1, 0, -1):
            dp[j] = triangle[i][j] + min(dp[j-1], dp[j])
        dp[0] = dp[0] + triangle[i][0]

    return min(dp)
```

### 3.5 C++ 实现

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    vector<int> dp = triangle[n-1];
    for (int i = n - 2; i >= 0; i--)
        for (int j = 0; j <= i; j++)
            dp[j] = triangle[i][j] + min(dp[j], dp[j+1]);
    return dp[0];
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 自顶向下 | O(n²) | O(n²) |
| 自底向上 | O(n²) | O(n) |
| 记忆化搜索 | O(n²) | O(n²) |

## 5. 典型例题

### 例题1：三角形最小路径和II（带路径还原）

```python
def minimumTotal_with_path(triangle):
    n = len(triangle)
    dp = [row[:] for row in triangle]

    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            dp[i][j] += min(dp[i+1][j], dp[i+1][j+1])

    # 还原路径
    path = [triangle[0][0]]
    j = 0
    for i in range(1, n):
        if dp[i][j] <= dp[i][j+1]:  # 优先走左边
            path.append(triangle[i][j])
        else:
            j += 1
            path.append(triangle[i][j])

    return dp[0][0], path
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **自顶向下边界处理**：j=0和j=i需要特殊处理
2. **自底向上更简洁**：不需要处理边界
3. **原地修改**：如果允许修改输入，可以用输入数组本身

### 6.2 为什么自底向上更好

```
自顶向下：
  - 需要处理第一列（只能从上一行同列来）
  - 需要处理对角线（只能从上一行前一列来）
  - 答案需要取最后一行的最小值

自底向上：
  - 每个位置都可以从下一行的两个相邻位置转移
  - 答案直接是 dp[0]
  - 代码更简洁
```
