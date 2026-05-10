# 解码方法 (Decode Ways)

## 1. 概念与定义

A-Z 编码为 1-26，给定一个只包含数字的字符串，求有多少种解码方式。

编码规则：
- 'A' -> 1, 'B' -> 2, ..., 'Z' -> 26
- 字符串中每个数字可以单独解码，也可以与后一个数字组合（如果在10~26范围内）

这是一个计数DP问题，与爬楼梯问题非常相似。

## 2. 状态定义与转移方程

```
dp[i] = 前i个字符的解码方法数
转移：
  if s[i-1] != '0': dp[i] += dp[i-1]  （单字符解码）
  if 10 <= int(s[i-2:i]) <= 26: dp[i] += dp[i-2]  （双字符解码）
dp[0] = 1（空串有一种解码方式）
dp[1] = 1 if s[0] != '0' else 0
```

### 注意特殊情况

- '0' 不能单独解码
- '00', '30', '40' 等不能双字符解码
- '0' 只能作为 '10' 或 '20' 的一部分

## 3. 算法实现

### 3.1 标准DP

```python
def numDecodings(s):
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        # 单字符
        if s[i - 1] != '0':
            dp[i] += dp[i - 1]
        # 双字符
        two_digit = int(s[i - 2:i])
        if 10 <= two_digit <= 26:
            dp[i] += dp[i - 2]

    return dp[n]
```

### 3.2 空间优化

```python
def numDecodings_optimized(s):
    if not s or s[0] == '0':
        return 0
    n = len(s)
    prev2, prev1 = 1, 1

    for i in range(2, n + 1):
        curr = 0
        if s[i - 1] != '0':
            curr += prev1
        if 10 <= int(s[i - 2:i]) <= 26:
            curr += prev2
        prev2, prev1 = prev1, curr

    return prev1
```

### 3.3 记忆化搜索

```python
from functools import lru_cache

def numDecodings_memo(s):
    @lru_cache(maxsize=None)
    def dfs(i):
        if i == len(s):
            return 1
        if s[i] == '0':
            return 0
        result = dfs(i + 1)  # 单字符
        if i + 1 < len(s) and 10 <= int(s[i:i+2]) <= 26:
            result += dfs(i + 2)  # 双字符
        return result

    return dfs(0)
```

### 3.4 带星号的解码（LeetCode 639）

```python
def numDecodings_with_star(s):
    MOD = 10**9 + 7
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        c = s[i - 1]
        if c == '*':
            # 单字符：* 可以是 1~9
            dp[i] = dp[i - 1] * 9
            # 双字符：和前一个字符组合
            if i >= 2:
                prev = s[i - 2]
                if prev == '1':
                    dp[i] += dp[i - 2] * 9  # 11~19
                elif prev == '2':
                    dp[i] += dp[i - 2] * 6  # 21~26
                elif prev == '*':
                    dp[i] += dp[i - 2] * 15  # 11~19 + 21~26
        else:
            if c != '0':
                dp[i] = dp[i - 1]
            if i >= 2:
                prev = s[i - 2]
                if prev == '*':
                    if int(c) <= 6:
                        dp[i] += dp[i - 2] * 2  # 1x, 2x
                    else:
                        dp[i] += dp[i - 2]  # 1x
                else:
                    two = int(s[i - 2:i])
                    if 10 <= two <= 26:
                        dp[i] += dp[i - 2]

        dp[i] %= MOD

    return dp[n]
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 标准DP | O(n) | O(n) |
| 空间优化 | O(n) | O(1) |
| 记忆化搜索 | O(n) | O(n) |

## 5. 典型例题

### 例题1：带星号的解码方法II（LeetCode 639）

```python
# 见上面 numDecodings_with_star 实现
```

### 例题2：解码方法II变形

```python
def numDecodings_with_zeros(s):
    """处理含0的情况"""
    if not s or s[0] == '0':
        return 0
    n = len(s)
    dp = [0] * (n + 1)
    dp[0] = 1
    dp[1] = 1

    for i in range(2, n + 1):
        if s[i-1] != '0':
            dp[i] += dp[i-1]
        two = int(s[i-2:i])
        if s[i-2] != '0' and 10 <= two <= 26:
            dp[i] += dp[i-2]

    return dp[n]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **以'0'开头**：直接返回0
2. **'0'的处理**：'0'不能单独解码，只能作为'10'或'20'
3. **连续'0'**：'00'无法解码
4. **'30'以上的'x0'**：'30','40'等无法解码

### 6.2 与爬楼梯的关系

```
爬楼梯：每步可以走1或2步，求到达第n阶的方法数
解码方法：每个位置可以取1或2个字符，求解码方案数

区别：
- 解码方法有额外限制（'0'不能单独、只能10~26等）
- 本质上是"有条件限制的爬楼梯"
```
