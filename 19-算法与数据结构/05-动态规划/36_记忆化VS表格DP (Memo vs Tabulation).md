# 记忆化VS表格DP (Memo vs Tabulation)

## 1. 概念与定义

动态规划有两种主要实现方式：
- **记忆化搜索（Memoization）**：自顶向下，递归 + 缓存
- **表格法（Tabulation）**：自底向上，迭代 + 数组

两者在时间复杂度上通常是等价的，但在代码风格、空间使用、适用场景上有区别。

## 2. 对比分析

### 2.1 方向对比

```
记忆化搜索（自顶向下）：
  solve(n) → solve(n-1), solve(n-2), ...
  从原问题出发，递归求解子问题

表格法（自底向上）：
  dp[0] → dp[1] → ... → dp[n]
  从最小的子问题出发，逐步推导到原问题
```

### 2.2 全面对比

| 特性 | 记忆化搜索 | 表格法 |
|------|-----------|--------|
| 方向 | 自顶向下 | 自底向上 |
| 实现 | 递归 + 缓存 | 迭代 + 数组 |
| 空间优化 | 较难 | 容易（滚动数组） |
| 栈溢出风险 | 有 | 无 |
| 子问题覆盖 | 只计算必要的 | 计算所有 |
| 代码可读性 | 更直观 | 需确定遍历顺序 |
| 调试难度 | 中等 | 较易 |
| 适用场景 | 转移复杂 | 转移简单 |

## 3. 算法实现

### 3.1 斐波那契数列

```python
from functools import lru_cache

# 记忆化搜索
@lru_cache(maxsize=None)
def fib_memo(n):
    if n <= 1:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)

# 表格法
def fib_tab(n):
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]
```

### 3.2 0-1背包

```python
# 记忆化搜索
def knapsack_memo(weights, values, capacity):
    n = len(weights)
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dfs(i, w):
        if i == n:
            return 0
        # 不选
        result = dfs(i + 1, w)
        # 选
        if w >= weights[i]:
            result = max(result, dfs(i + 1, w - weights[i]) + values[i])
        return result
    return dfs(0, capacity)

# 表格法
def knapsack_tab(weights, values, capacity):
    n = len(weights)
    dp = [0] * (capacity + 1)
    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])
    return dp[capacity]
```

### 3.3 编辑距离

```python
# 记忆化搜索
def editDistance_memo(s1, s2):
    from functools import lru_cache
    @lru_cache(maxsize=None)
    def dp(i, j):
        if i < 0: return j + 1
        if j < 0: return i + 1
        if s1[i] == s2[j]:
            return dp(i-1, j-1)
        return 1 + min(dp(i-1,j), dp(i,j-1), dp(i-1,j-1))
    return dp(len(s1)-1, len(s2)-1)

# 表格法
def editDistance_tab(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j
    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]
```

## 4. 何时选择哪种方式

### 4.1 选记忆化搜索

- 转移方程复杂，不容易确定遍历顺序
- 不是所有状态都需要计算（稀疏状态空间）
- 问题的递归结构非常自然
- 快速原型开发（先写出正确解）

### 4.2 选表格法

- 需要空间优化（滚动数组）
- 状态空间密集，所有状态都需要计算
- 递归深度可能很大
- 性能要求高（避免递归调用开销）

### 4.3 实际建议

```
1. 先用记忆化搜索快速写出正确解
2. 验证正确性
3. 如果需要优化，再改写为表格法
4. 如果需要空间优化，使用滚动数组
```

## 5. 复杂度分析

### 5.1 时间复杂度

两者计算的子问题数量相同，因此时间复杂度一致。

### 5.2 空间复杂度

- 记忆化搜索：O(状态数) + O(递归栈深度)
- 表格法：O(状态数)，可优化到 O(滚动行数)

### 5.3 常数因子

- 记忆化搜索：函数调用开销
- 表格法：循环开销较小
- 在极端优化下表格法略快

## 6. 常见陷阱

### 6.1 记忆化搜索的陷阱

1. 递归深度超限（Python默认约1000）
2. 可变默认参数
3. 忘记添加缓存
4. lru_cache 的 self 泄漏

### 6.2 表格法的陷阱

1. 遍历顺序错误
2. 边界条件遗漏
3. 0-1背包正序遍历
