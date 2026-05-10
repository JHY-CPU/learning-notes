# 整数拆分 (Integer Break)

## 1. 概念与定义

整数拆分问题：给定一个正整数 n，将其拆分为至少两个正整数的和，使这些整数的乘积最大。

核心发现：
- 尽可能多地拆出3：当 n >= 5 时，3(n-3) >= n
- 余数处理：n % 3 == 0 时全用3；n % 3 == 1 时拿出一个3变成两个2；n % 3 == 2 时用一个2
- 数学证明：对于 n >= 5，拆分出的因子不应超过3

## 2. 状态定义与转移方程

### 2.1 DP方法

```
dp[i] = 将正整数i拆分成至少两个正整数的和的最大乘积
dp[i] = max(j * max(i-j, dp[i-j])) for j in [1, i-1]
dp[1] = 1
注意：j * (i-j) 是只拆成两部分，j * dp[i-j] 是继续拆分第二部分
```

### 2.2 数学方法

```
如果 n % 3 == 0: answer = 3^(n/3)
如果 n % 3 == 1: answer = 3^(n/3-1) * 4  (因为3*1 < 2*2)
如果 n % 3 == 2: answer = 3^(n/3) * 2
特例：n <= 3 时，answer = n-1
```

## 3. 算法实现

### 3.1 DP方法

```python
def integerBreak(n):
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(1, i):
            dp[i] = max(dp[i], j * (i - j), j * dp[i - j])
    return dp[n]
```

### 3.2 数学方法

```python
def integerBreak_math(n):
    if n <= 3:
        return n - 1
    a, b = divmod(n, 3)
    if b == 0:
        return 3 ** a
    elif b == 1:
        return 3 ** (a - 1) * 4
    else:
        return 3 ** a * 2
```

### 3.3 C++ 实现

```cpp
int integerBreak(int n) {
    vector<int> dp(n + 1, 0);
    dp[1] = 1;
    for (int i = 2; i <= n; i++)
        for (int j = 1; j < i; j++)
            dp[i] = max({dp[i], j * (i - j), j * dp[i - j]});
    return dp[n];
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| DP | O(n²) | O(n) |
| 数学 | O(1) | O(1) |

## 5. 典型例题

### 例题1：不同的二叉搜索树（LeetCode 96）

```python
def numTrees(n):
    """卡特兰数"""
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    return dp[n]
```

### 例题2：整数拆分的因子个数（LeetCode 343变体）

```python
def maxProductWithFactors(n):
    """返回最大乘积及对应的因子"""
    dp = [0] * (n + 1)
    factors = [[] for _ in range(n + 1)]
    dp[1] = 1

    for i in range(2, n + 1):
        for j in range(1, i):
            if j * (i - j) > dp[i]:
                dp[i] = j * (i - j)
                factors[i] = [j, i - j]
            if j * dp[i - j] > dp[i]:
                dp[i] = j * dp[i - j]
                factors[i] = [j] + factors[i - j]

    return dp[n], factors[n]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **至少两个正整数**：n=2,3时答案不是n，而是n-1
2. **dp[i-j] vs (i-j)**：`j * dp[i-j]` 是继续拆分，`j * (i-j)` 是不继续拆
3. **特例处理**：n <= 3 时数学公式需要特殊处理

### 6.2 数学证明

为什么因子尽量是3？
- 对于 k >= 5：3(k-3) = 3k-9 > k（拆成3和k-3更好）
- 对于 k = 4：2*2 = 4 = 4（拆不拆都一样）
- 对于 k = 3：不拆更好
- 所以当 n >= 5 时，不断拆出3是最优的
