# 布尔表达式DP (Boolean Parenthesization)

## 1. 概念与定义

布尔表达式求值问题：给定一个由 True/False 和运算符（&, |, ^）组成的布尔表达式，通过添加括号改变求值顺序，求使表达式结果为 True 的括号方案数。

这是一个区间DP问题。对于区间 [i, j]，枚举每个运算符作为最后执行的操作，将表达式分成左右两部分。

## 2. 状态定义与转移方程

```
dp_true[i][j] = 区间[i,j]的表达式能求出True的方案数
dp_false[i][j] = 区间[i,j]的表达式能求出False的方案数

转移（枚举运算符k）：
  if op[k] == '&':
    dp_true[i][j] += dp_true[i][k] * dp_true[k+1][j]
    dp_false[i][j] += dp_true[i][k]*dp_false[k+1][j] + dp_false[i][k]*dp_true[k+1][j] + dp_false[i][k]*dp_false[k+1][j]

  if op[k] == '|':
    dp_false[i][j] += dp_false[i][k] * dp_false[k+1][j]
    dp_true[i][j] += dp_true[i][k]*dp_true[k+1][j] + dp_true[i][k]*dp_false[k+1][j] + dp_false[i][k]*dp_true[k+1][j]

  if op[k] == '^':
    dp_true[i][j] += dp_true[i][k]*dp_false[k+1][j] + dp_false[i][k]*dp_true[k+1][j]
    dp_false[i][j] += dp_true[i][k]*dp_true[k+1][j] + dp_false[i][k]*dp_false[k+1][j]

初始条件：
  dp_true[i][i] = 1 if values[i] == True else 0
  dp_false[i][i] = 1 if values[i] == False else 0
```

## 3. 算法实现

### 3.1 标准区间DP

```python
def countEval(s, goal):
    """
    s: 布尔表达式字符串，如 "T|F&T^T"
    goal: 目标结果，True 或 False
    """
    n = len(s)
    # 提取值和运算符
    values = []
    ops = []
    for i, c in enumerate(s):
        if c in 'TF':
            values.append(c == 'T')
        else:
            ops.append(c)

    m = len(values)  # 值的个数
    dp_true = [[0] * m for _ in range(m)]
    dp_false = [[0] * m for _ in range(m)]

    # 初始化
    for i in range(m):
        if values[i]:
            dp_true[i][i] = 1
        else:
            dp_false[i][i] = 1

    # 区间DP
    for length in range(2, m + 1):
        for i in range(m - length + 1):
            j = i + length - 1
            for k in range(i, j):
                op = ops[k]
                total_true_left = dp_true[i][k]
                total_false_left = dp_false[i][k]
                total_true_right = dp_true[k + 1][j]
                total_false_right = dp_false[k + 1][j]
                total_left = total_true_left + total_false_left
                total_right = total_true_right + total_false_right

                if op == '&':
                    dp_true[i][j] += total_true_left * total_true_right
                    dp_false[i][j] += total_left * total_right - total_true_left * total_true_right
                elif op == '|':
                    dp_false[i][j] += total_false_left * total_false_right
                    dp_true[i][j] += total_left * total_right - total_false_left * total_false_right
                elif op == '^':
                    dp_true[i][j] += total_true_left * total_false_right + total_false_left * total_true_right
                    dp_false[i][j] += total_true_left * total_true_right + total_false_left * total_false_right

    return dp_true[0][m - 1] if goal else dp_false[0][m - 1]
```

### 3.2 记忆化搜索

```python
from functools import lru_cache

def countEval_memo(s, goal):
    values = [c == 'T' for c in s if c in 'TF']
    ops = [c for c in s if c in '&|^']

    @lru_cache(maxsize=None)
    def dp(i, j):
        """返回 (true_count, false_count) for 区间[i,j]"""
        if i == j:
            return (1, 0) if values[i] else (0, 1)

        true_cnt, false_cnt = 0, 0
        for k in range(i, j):
            lt, lf = dp(i, k)
            rt, rf = dp(k + 1, j)

            if ops[k] == '&':
                true_cnt += lt * rt
                false_cnt += lt * rf + lf * rt + lf * rf
            elif ops[k] == '|':
                false_cnt += lf * rf
                true_cnt += lt * rt + lt * rf + lf * rt
            elif ops[k] == '^':
                true_cnt += lt * rf + lf * rt
                false_cnt += lt * rt + lf * rf

        return (true_cnt, false_cnt)

    return dp(0, len(values) - 1)[0 if goal else 1]
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 区间DP | O(n³) | O(n²) |
| 记忆化搜索 | O(n³) | O(n²) |

其中 n 为布尔值的个数。

## 5. 典型例题

### 例题1：布尔表达式求值（LeetCode 1106）

```python
def parseBoolExpr(expression):
    """解析并求值布尔表达式"""
    stack = []
    for c in expression:
        if c == ')':
            vals = []
            while stack[-1] != '(':
                vals.append(stack.pop())
            stack.pop()  # 弹出 '('
            op = stack.pop()
            if op == '!':
                stack.append('t' if vals[0] == 'f' else 'f')
            elif op == '&':
                stack.append('t' if all(v == 't' for v in vals) else 'f')
            elif op == '|':
                stack.append('t' if any(v == 't' for v in vals) else 'f')
        elif c != ',':
            stack.append(c)

    return stack[0] == 't'
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **运算符数量 = 值数量 - 1**
2. **异或运算**：结果与两边相同/不同相关
3. **方案数可能很大**：需要取模
4. **区间分割**：k 是运算符的索引，对应值区间 [i,k] 和 [k+1,j]

### 6.2 运算真值表

```
& (AND): T&T=T, T&F=F, F&T=F, F&F=F
| (OR):  T|T=T, T|T=T, F|T=T, F|F=F
^ (XOR): T^T=F, T^F=T, F^T=T, F^F=F
```
