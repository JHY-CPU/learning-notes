# 区间DP (Interval DP)

## 1. 概念与定义

区间DP是一类以区间为状态的动态规划。通常用 `dp[i][j]` 表示区间 [i, j] 上的最优值。核心思想：**小区间的最优解可以合并为大区间的最优解**。

区间DP的基本特征：
- 问题涉及**连续区间**的操作
- 大区间可以由小区间**合并/拼接**得到
- 通常需要枚举**分割点** k

经典问题：石子合并、矩阵链乘法、戳气球

## 2. 状态定义与转移方程

### 2.1 通用框架

```
dp[i][j] = 区间 [i, j] 的最优值
dp[i][j] = min/max(dp[i][k] + dp[k+1][j] + cost(i, j, k))  for k in [i, j-1]

枚举顺序：按区间长度从小到大
for len in range(2, n+1):
    for i in range(0, n-len+1):
        j = i + len - 1
        for k in range(i, j):
            dp[i][j] = ...
```

### 2.2 石子合并
```
dp[i][j] = min(dp[i][k] + dp[k+1][j]) + sum(i, j)
dp[i][i] = 0
```

### 2.3 矩阵链乘法
```
dp[i][j] = min(dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j])
dp[i][i] = 0
```

### 2.4 戳气球
```
dp[i][j] = max(dp[i][k] + dp[k][j] + nums[i]*nums[k]*nums[j])
```

## 3. 算法实现

### 3.1 矩阵链乘法

```python
def matrixChainOrder(p):
    n = len(p) - 1
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')
            for k in range(i, j):
                cost = dp[i][k] + dp[k + 1][j] + p[i - 1] * p[k] * p[j]
                dp[i][j] = min(dp[i][j], cost)
    return dp[1][n]
```

### 3.2 戳气球（LeetCode 312）

```python
def maxCoins(nums):
    nums = [1] + nums + [1]
    n = len(nums)
    dp = [[0] * n for _ in range(n)]
    for length in range(2, n):
        for left in range(n - length):
            right = left + length
            for k in range(left + 1, right):
                coins = nums[left] * nums[k] * nums[right]
                dp[left][right] = max(dp[left][right],
                                      dp[left][k] + dp[k][right] + coins)
    return dp[0][n - 1]
```

### 3.3 最长回文子序列（LeetCode 516）

```python
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 1
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])
    return dp[0][n - 1]
```

### 3.4 C++ 实现

```cpp
int matrixChain(vector<int>& p) {
    int n = p.size() - 1;
    vector<vector<int>> dp(n+1, vector<int>(n+1, 0));
    for (int len = 2; len <= n; len++)
        for (int i = 1; i <= n-len+1; i++) {
            int j = i + len - 1;
            dp[i][j] = INT_MAX;
            for (int k = i; k < j; k++)
                dp[i][j] = min(dp[i][j], dp[i][k] + dp[k+1][j] + p[i-1]*p[k]*p[j]);
        }
    return dp[1][n];
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 石子合并 | O(n³) | O(n²) |
| 矩阵链乘法 | O(n³) | O(n²) |
| 戳气球 | O(n³) | O(n²) |

## 5. 典型例题

### 例题1：最小得分三角剖分（LeetCode 1039）

```python
def minScoreTriangulation(values):
    n = len(values)
    dp = [[0] * n for _ in range(n)]
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = (i + length - 1) % n
            dp[i][j] = float('inf')
            for k in range(i + 1, i + length - 1):
                k_mod = k % n
                dp[i][j] = min(dp[i][j],
                    dp[i][k_mod] + dp[k_mod][j] + values[i] * values[k_mod] * values[j])
    return dp[0][n - 1]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **枚举顺序错误**：必须按区间长度从小到大枚举
2. **分割点范围**：k 从 i 到 j-1
3. **环形处理**：将数组复制一遍或用取模
4. **cost函数**：可能需要前缀和预处理

### 6.2 区间DP模板

```python
def interval_dp(arr):
    n = len(arr)
    dp = [[0] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 初始值
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = 极值
            for k in range(i, j):
                dp[i][j] = 最优(dp[i][j], dp[i][k] + dp[k+1][j] + cost)
    return dp[0][n - 1]
```

### 6.3 识别区间DP的特征

- 涉及**合并**操作
- 答案依赖于**区间的合并方式**
- 子问题之间有**相邻关系**
- 求一个区间上的**最优值**
