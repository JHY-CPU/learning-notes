# 计数DP (Counting DP)

## 1. 概念与定义

计数DP是用于**统计满足特定条件的方案数**的动态规划。与最值DP不同，计数DP的转移操作是加法而不是取最大/最小值。

计数DP的核心公式：
```
dp[i] = Σ dp[j]  对所有能转移到i的状态j
```

常见应用：
- 路径计数
- 组合计数
- 排列计数
- 整数划分
- 格路问题

## 2. 状态定义与转移方程

### 2.1 路径计数

```
dp[i][j] = 从(0,0)到(i,j)的路径数
dp[i][j] = dp[i-1][j] + dp[i][j-1]
dp[0][0] = 1, dp[i][0] = 1, dp[0][j] = 1
```

### 2.2 组合计数（递推公式）

```
C(n, k) = C(n-1, k-1) + C(n-1, k)
C(n, 0) = C(n, n) = 1
```

### 2.3 整数划分

```
dp[i][j] = 用前i个正整数表示j的方案数
dp[i][j] = dp[i-1][j] + dp[i][j-i]  （不选i + 选i）
dp[0][0] = 1
```

### 2.4 卡特兰数

```
C(n) = C(0)*C(n-1) + C(1)*C(n-2) + ... + C(n-1)*C(0)
C(0) = 1
通项：C(n) = C(2n, n) / (n+1)
```

## 3. 算法实现

### 3.1 组合数取模

```python
MOD = 10**9 + 7

def comb_table(n, k):
    """打表法求组合数 C(n, k)"""
    C = [[0] * (k + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        C[i][0] = 1
        for j in range(1, min(i, k) + 1):
            C[i][j] = (C[i - 1][j - 1] + C[i - 1][j]) % MOD
    return C[n][k]

def comb_fact(n, k):
    """阶乘法求组合数（适合大量查询）"""
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % MOD

    def pow_mod(a, b):
        result = 1
        while b:
            if b & 1:
                result = result * a % MOD
            a = a * a % MOD
            b >>= 1
        return result

    inv_fact = [1] * (n + 1)
    inv_fact[n] = pow_mod(fact[n], MOD - 2)
    for i in range(n - 1, -1, -1):
        inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

    return fact[n] * inv_fact[k] % MOD * inv_fact[n - k] % MOD
```

### 3.2 不同路径（LeetCode 62）

```python
def uniquePaths(m, n):
    """网格路径计数"""
    from math import comb
    return comb(m + n - 2, m - 1)
```

### 3.3 卡特兰数

```python
def catalan(n):
    """卡特兰数 C(n) = C(2n, n) / (n+1)"""
    from math import comb
    return comb(2 * n, n) // (n + 1)

def catalan_dp(n):
    """DP方式求卡特兰数"""
    dp = [0] * (n + 1)
    dp[0] = 1
    for i in range(1, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    return dp[n]
```

### 3.4 不同的二叉搜索树（LeetCode 96）

```python
def numTrees(n):
    """n个节点能组成多少种BST = 卡特兰数"""
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1
    for i in range(2, n + 1):
        for j in range(i):
            dp[i] += dp[j] * dp[i - 1 - j]
    return dp[n]
```

### 3.5 有效括号对数（LeetCode 22）

```python
def generateParenthesis(n):
    """生成所有有效括号组合（非DP，但基于卡特兰数）"""
    result = []
    def backtrack(curr, open_count, close_count):
        if len(curr) == 2 * n:
            result.append(curr)
            return
        if open_count < n:
            backtrack(curr + '(', open_count + 1, close_count)
        if close_count < open_count:
            backtrack(curr + ')', open_count, close_count + 1)
    backtrack('', 0, 0)
    return result
```

### 3.6 排序序列数（LeetCode 903）

```python
def numPermsDISequence(s):
    """统计DI序列的有效排列数"""
    MOD = 10**9 + 7
    n = len(s)
    # dp[i][j] = 使用0~i的数字，以j结尾的方案数
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(i + 1):
            if s[i - 1] == 'I':
                for k in range(j):
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD
            else:
                for k in range(j, i):
                    dp[i][j] = (dp[i][j] + dp[i - 1][k]) % MOD

    return sum(dp[n]) % MOD
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 组合数（打表） | O(nk) | O(nk) |
| 组合数（阶乘） | O(n) 预处理 | O(n) |
| 卡特兰数（DP） | O(n²) | O(n) |
| 路径计数 | O(mn) | O(mn)或O(n) |

## 5. 典型例题

### 例题1：统计元音字符串（LeetCode 2550）

```python
def countVowelStrings(n):
    """长度为n的、元音字母非递减排列的字符串个数"""
    # dp[i][j] = 长度为i、以第j个元音结尾的方案数
    dp = [[0] * 5 for _ in range(n + 1)]
    for j in range(5):
        dp[1][j] = 1
    for i in range(2, n + 1):
        for j in range(5):
            dp[i][j] = sum(dp[i - 1][:j + 1])
    return sum(dp[n])
```

### 例题2：统计好子集（LeetCode 1994）

```python
def numberOfGoodSubsets(nums):
    """统计乘积为无平方因子数的非空子集数"""
    MOD = 10**9 + 7
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    from collections import Counter
    count = Counter(nums)

    # dp[mask] = 质数集合为mask的方案数
    dp = [0] * (1 << 10)
    dp[0] = 1

    for num in range(2, 31):
        if count[num] == 0:
            continue
        # 检查是否有平方因子
        x, mask = num, 0
        ok = True
        for i, p in enumerate(primes):
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt > 1:
                ok = False
                break
            if cnt == 1:
                mask |= (1 << i)
        if not ok or x > 1:
            continue
        # 更新dp
        for s in range((1 << 10) - 1, -1, -1):
            if s & mask == 0:
                dp[s | mask] = (dp[s | mask] + dp[s] * count[num]) % MOD

    # 1可以选或不选
    return sum(dp[1:]) % MOD * pow(2, count[1], MOD) % MOD
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **取模问题**：大数需要取模，注意减法取模要加MOD
2. **初始化**：dp[0] 通常是 1（空方案）
3. **组合vs排列**：注意问题是否考虑顺序
4. **整数溢出**：Python不会溢出，但C/C++需要注意

### 6.2 常用组合恒等式

```
C(n, k) = C(n, n-k)
C(n, k) = C(n-1, k-1) + C(n-1, k)
C(n, 0) + C(n, 1) + ... + C(n, n) = 2^n
卡特兰数：C(n) = C(2n, n) / (n+1)
```

### 6.3 优化技巧

1. **前缀和优化**：当转移涉及区间和时
2. **滚动数组**：减少空间
3. **数学公式**：某些计数有闭式公式
