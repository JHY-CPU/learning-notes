# 四边形不等式优化 (Quadrangle Inequality)

## 1. 概念与定义

四边形不等式优化是一种将区间DP的时间复杂度从 **O(n³)** 优化到 **O(n²)** 的技巧。

核心概念：

**区间包含单调性**：若 w(i, j) 满足对于任意 a <= b <= c <= d，有：
```
w(b, c) <= w(a, d)
```

**四边形不等式**：若对于任意 a <= b <= c <= d，有：
```
w(a, c) + w(b, d) <= w(a, d) + w(b, c)
```

当 w 满足四边形不等式时，令 s[i][j] 为 dp[i][j] 的最优分割点，有：
```
s[i][j-1] <= s[i][j] <= s[i+1][j]
```

这使得搜索空间大大缩小，时间复杂度从 O(n³) 降到 O(n²)。

## 2. 状态定义与转移方程

### 2.1 区间DP原始形式

```
dp[i][j] = min(dp[i][k] + dp[k+1][j] + w(i, j))  for k in [i, j-1]
时间复杂度：O(n³)
```

### 2.2 四边形不等式优化后

```
dp[i][j] = min(dp[i][k] + dp[k+1][j] + w(i, j))  for k in [s[i][j-1], s[i+1][j]]
时间复杂度：O(n²)

枚举顺序：
for i in range(n): dp[i][i] = 0, s[i][i] = i
for length in range(2, n+1):
    for i in range(n - length + 1):
        j = i + length - 1
        for k in range(s[i][j-1], s[i+1][j] + 1):
            ...
```

### 2.3 适用条件

代价函数 w(i, j) 需要满足：
1. 区间包含单调性
2. 四边形不等式

常见满足条件的 w：
- w(i, j) = sum(i, j)（区间和）
- w(i, j) = j - i + 1（区间长度）
- w(i, j) = (j - i) * something

## 3. 算法实现

### 3.1 石子合并优化

```python
def mergeStones_optimized(stones):
    """四边形不等式优化的石子合并"""
    n = len(stones)
    # 前缀和
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + stones[i]

    def w(i, j):
        return prefix[j + 1] - prefix[i]

    INF = float('inf')
    dp = [[0] * n for _ in range(n)]
    s = [[0] * n for _ in range(n)]

    for i in range(n):
        s[i][i] = i

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = INF
            # 只在 [s[i][j-1], s[i+1][j]] 范围内枚举
            left = s[i][j - 1]
            right = s[i + 1][j] if i + 1 <= j else j - 1
            for k in range(left, right + 1):
                cost = dp[i][k] + dp[k + 1][j] + w(i, j)
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    s[i][j] = k

    return dp[0][n - 1]
```

### 3.2 矩阵链乘法优化

```python
def matrixChain_optimized(p):
    n = len(p) - 1
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    s = [[0] * (n + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        s[i][i] = i

    for length in range(2, n + 1):
        for i in range(1, n - length + 2):
            j = i + length - 1
            dp[i][j] = float('inf')
            left = s[i][j - 1]
            right = s[i + 1][j]
            for k in range(left, right + 1):
                cost = dp[i][k] + dp[k + 1][j] + p[i-1] * p[k] * p[j]
                if cost < dp[i][j]:
                    dp[i][j] = cost
                    s[i][j] = k

    return dp[1][n]
```

### 3.3 判断四边形不等式

```python
def checkQuadrangle(w, n):
    """验证w是否满足四边形不等式"""
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(b, n):
                for d in range(c, n):
                    if w(a, c) + w(b, d) > w(a, d) + w(b, c):
                        return False
    return True
```

## 4. 复杂度分析

| 问题 | 原始复杂度 | 优化后复杂度 |
|------|-----------|-------------|
| 石子合并 | O(n³) | O(n²) |
| 矩阵链乘法 | O(n³) | O(n²) |
| 最优BST | O(n³) | O(n²) |

优化效果：n=1000 时，O(n³) = 10^9 次操作，O(n²) = 10^6 次操作。

## 5. 典型例题

### 例题1：邮局选址问题

```
在一条直线上有n个村庄，要建p个邮局。
求每个村庄到最近邮局的距离之和的最小值。
```

```python
def postOffice(villages, p):
    """邮局问题：四边形不等式优化"""
    villages.sort()
    n = len(villages)

    # 预处理 cost[i][j]：在 villages[i..j] 之间建1个邮局的最小距离和
    cost = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            mid = (i + j) // 2
            for k in range(i, j + 1):
                cost[i][j] += abs(villages[k] - villages[mid])

    # dp[i][j] = 在前i个村庄建j个邮局的最小距离和
    INF = float('inf')
    dp = [[INF] * (p + 1) for _ in range(n)]
    s = [[0] * (p + 1) for _ in range(n)]

    for i in range(n):
        dp[i][1] = cost[0][i]
        s[i][1] = 0

    for j in range(2, p + 1):
        s[n - 1][j] = n - 2
        for i in range(n - 1, j - 2, -1):
            dp[i][j] = INF
            left = s[i][j - 1] if i > 0 else 0
            right = s[i + 1][j] if i + 1 < n else i - 1
            for k in range(left, min(right, i - 1) + 1):
                val = dp[k][j - 1] + cost[k + 1][i]
                if val < dp[i][j]:
                    dp[i][j] = val
                    s[i][j] = k

    return dp[n - 1][p]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **不验证四边形不等式**：不是所有区间DP都满足
2. **边界处理**：s[i][j-1] 和 s[i+1][j] 的范围需要正确处理
3. **枚举顺序**：length 从小到大，确保 s[i][j-1] 和 s[i+1][j] 已计算
4. **代价函数**：w 必须是区间包含单调且满足四边形不等式

### 6.2 常见满足四边形不等式的函数

```
w(i, j) = Σ a[k]  (k从i到j) — 区间和
w(i, j) = max(a[k]) (k从i到j) — 区间最大值
w(i, j) = j - i — 区间长度
w(i, j) = (prefix[j+1] - prefix[i])² — 前缀和平方
```

### 6.3 识别时机

- 区间DP的O(n³)过不了
- 代价函数看起来满足四边形不等式
- 分割点有单调性质
