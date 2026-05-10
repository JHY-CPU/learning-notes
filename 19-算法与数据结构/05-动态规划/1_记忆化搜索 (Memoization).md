# 记忆化搜索 (Memoization)

## 1. 概念与定义

记忆化搜索（Memoization）是动态规划的一种实现方式，采用**自顶向下**（Top-Down）的策略。其核心思想是在递归过程中，将已经计算过的结果保存到缓存中，当再次遇到相同的子问题时直接返回缓存结果，避免重复计算。

记忆化搜索的本质是：**递归 + 缓存 = 动态规划**

与自底向上的表格法（Tabulation）相比：
- 记忆化搜索从原问题出发，递归地求解子问题
- 表格法从最小的子问题出发，逐步推导到原问题
- 两者在时间复杂度上通常是等价的
- 记忆化搜索只计算必要的子问题，而表格法计算所有子问题

## 2. 状态定义与转移方程

### 2.1 记忆化搜索的框架

```python
def dp(状态参数):
    # 1. 基础情况（Base Case）
    if 边界条件:
        return 基础值

    # 2. 查缓存
    if 状态 in memo:
        return memo[状态]

    # 3. 递归计算
    result = 0
    for 选择 in 所有可能的选择:
        result = 最优(result, dp(新状态))

    # 4. 存入缓存并返回
    memo[状态] = result
    return result
```

### 2.2 以斐波那契数列为例

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
```

等价的手动记忆化版本：

```python
def fib_memo(n, memo={}):
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo)
    return memo[n]
```

## 3. 算法实现

### 3.1 Python lru_cache 装饰器

```python
from functools import lru_cache

@lru_cache(maxsize=None)
def dp(i, j):
    if i == 0 or j == 0:
        return 0
    return dp(i-1, j) + dp(i, j-1)

# 使用完毕后清缓存
dp.cache_clear()
```

### 3.2 手动记忆化实现

```python
def knapsack_memo(weights, values, capacity):
    """记忆化搜索实现0-1背包"""
    n = len(weights)
    memo = {}

    def dfs(i, remaining):
        if i == n or remaining == 0:
            return 0
        if (i, remaining) in memo:
            return memo[(i, remaining)]

        result = dfs(i + 1, remaining)
        if remaining >= weights[i]:
            result = max(result, dfs(i + 1, remaining - weights[i]) + values[i])

        memo[(i, remaining)] = result
        return result

    return dfs(0, capacity)
```

### 3.3 C++ 实现

```cpp
#include <bits/stdc++.h>
using namespace std;

int dp[1001][1001];
bool vis[1001][1001];

int dfs(int i, int j, vector<int>& w, vector<int>& v) {
    if (i < 0) return 0;
    if (vis[i][j]) return dp[i][j];
    vis[i][j] = true;
    dp[i][j] = dfs(i - 1, j, w, v);
    if (j >= w[i]) {
        dp[i][j] = max(dp[i][j], dfs(i - 1, j - w[i], w, v) + v[i]);
    }
    return dp[i][j];
}
```

### 3.4 编辑距离的记忆化实现

```python
from functools import lru_cache

def edit_distance(s1, s2):
    @lru_cache(maxsize=None)
    def dp(i, j):
        if i < 0:
            return j + 1
        if j < 0:
            return i + 1
        if s1[i] == s2[j]:
            return dp(i - 1, j - 1)
        return min(
            dp(i - 1, j) + 1,
            dp(i, j - 1) + 1,
            dp(i - 1, j - 1) + 1
        )

    return dp(len(s1) - 1, len(s2) - 1)
```

## 4. 复杂度分析

### 4.1 与表格法的对比

| 特性 | 记忆化搜索 | 表格法 |
|------|-----------|--------|
| 方向 | 自顶向下 | 自底向上 |
| 实现方式 | 递归+缓存 | 迭代+数组 |
| 空间优化 | 较难 | 较易（滚动数组） |
| 栈溢出风险 | 有 | 无 |
| 子问题覆盖 | 只计算必要的 | 计算所有 |
| 代码可读性 | 更直观 | 需要确定遍历顺序 |

## 5. 典型例题

### 例题1：零钱兑换（LeetCode 322）

```python
from functools import lru_cache

def coinChange(coins, amount):
    @lru_cache(maxsize=None)
    def dp(remain):
        if remain == 0:
            return 0
        if remain < 0:
            return float('inf')
        result = float('inf')
        for coin in coins:
            sub = dp(remain - coin)
            if sub != float('inf'):
                result = min(result, sub + 1)
        return result

    ans = dp(amount)
    return ans if ans != float('inf') else -1
```

### 例题2：不同路径 II（LeetCode 63）

```python
from functools import lru_cache

def uniquePathsWithObstacles(grid):
    m, n = len(grid), len(grid[0])

    @lru_cache(maxsize=None)
    def dp(i, j):
        if i < 0 or j < 0:
            return 0
        if grid[i][j] == 1:
            return 0
        if i == 0 and j == 0:
            return 1
        return dp(i - 1, j) + dp(i, j - 1)

    return dp(m - 1, n - 1)
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **忘记添加缓存**：写成了纯递归，导致超时
2. **可变对象作为缓存键**：使用列表作为字典的key会报错
3. **Python默认参数陷阱**：

```python
# 错误！默认参数是可变对象
def dp(i, memo={}):
    ...

# 正确
def dp(i, memo=None):
    if memo is None:
        memo = {}
    ...
```

4. **递归深度超限**：Python默认递归深度约1000
5. **缓存未清理**：多次调用时缓存保留旧结果

### 6.2 lru_cache 注意事项

- 参数必须是**可哈希的**
- 使用 `@lru_cache(None)` 不限制缓存大小
- 使用 `fn.cache_info()` 查看缓存命中率
- 使用 `fn.cache_clear()` 清除缓存
