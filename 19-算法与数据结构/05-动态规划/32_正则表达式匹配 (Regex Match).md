# 正则表达式匹配 (Regex Match)

## 1. 概念与定义

正则表达式匹配问题：给定字符串 s 和模式 p，判断 s 是否匹配 p。模式支持：
- `.` 匹配任意单个字符
- `*` 匹配零个或多个前面的元素

这是一个经典的二维DP问题，需要仔细处理 `*` 的特殊含义。

## 2. 状态定义与转移方程

```
dp[i][j] = s[0..i-1] 是否匹配 p[0..j-1]

转移：
  if p[j-1] == '*':
    # * 匹配零个前面的元素：dp[i][j] = dp[i][j-2]
    # * 匹配一个或多个：dp[i][j] = dp[i-1][j] 且 s[i-1] 能与 p[j-2] 匹配
    dp[i][j] = dp[i][j-2] or (dp[i-1][j] and match(s[i-1], p[j-2]))
  else:
    dp[i][j] = dp[i-1][j-1] and match(s[i-1], p[j-1])

初始条件：
  dp[0][0] = True
  dp[0][j] = dp[0][j-2] if p[j-1] == '*'  （空串可以被x*匹配）
```

## 3. 算法实现

### 3.1 标准DP（LeetCode 10）

```python
def isMatch(s, p):
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # 初始化：空串与模式的匹配
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # 零个前面的元素
                dp[i][j] = dp[i][j - 2]
                # 一个或多个前面的元素
                if p[j - 2] == '.' or p[j - 2] == s[i - 1]:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            else:
                if p[j - 1] == '.' or p[j - 1] == s[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]
```

### 3.2 通配符匹配（LeetCode 44）

```python
def isMatch_wildcard(s, p):
    """
    ? 匹配任意单个字符
    * 匹配任意字符串（包括空串）
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    # 空串：只有全是 * 的模式才能匹配
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # * 匹配空串：dp[i][j-1]
                # * 匹配一个或多个字符：dp[i-1][j]
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]
```

### 3.3 空间优化

```python
def isMatch_optimized(s, p):
    m, n = len(s), len(p)
    dp = [False] * (n + 1)
    dp[0] = True

    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[j] = dp[j - 2]

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = False
        for j in range(1, n + 1):
            temp = dp[j]
            if p[j - 1] == '*':
                dp[j] = dp[j - 2] or (dp[j] and (p[j - 2] == '.' or p[j - 2] == s[i - 1]))
            else:
                dp[j] = prev and (p[j - 1] == '.' or p[j - 1] == s[i - 1])
            prev = temp

    return dp[n]
```

### 3.4 C++ 实现

```cpp
bool isMatch(string s, string p) {
    int m = s.size(), n = p.size();
    vector<vector<bool>> dp(m+1, vector<bool>(n+1, false));
    dp[0][0] = true;
    for (int j = 2; j <= n; j++)
        if (p[j-1] == '*') dp[0][j] = dp[0][j-2];
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++) {
            if (p[j-1] == '*') {
                dp[i][j] = dp[i][j-2];
                if (p[j-2] == '.' || p[j-2] == s[i-1])
                    dp[i][j] = dp[i][j] || dp[i-1][j];
            } else if (p[j-1] == '.' || p[j-1] == s[i-1]) {
                dp[i][j] = dp[i-1][j-1];
            }
        }
    return dp[m][n];
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 标准DP | O(mn) | O(mn) |
| 空间优化 | O(mn) | O(n) |
| 递归（无记忆） | O(2^(m+n)) | O(m+n) |

## 5. 典型例题

### 例题1：正则表达式匹配（LeetCode 10）

```python
# 见 isMatch 实现
```

### 例题2：通配符匹配（LeetCode 44）

```python
# 见 isMatch_wildcard 实现
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **`*` 的含义不同**：正则表达式中 `*` 匹配零个或多个**前面的元素**；通配符中 `*` 匹配任意字符串
2. **`*` 不能单独出现**：模式中 `*` 前面必须有字符
3. **空串处理**：空串与模式的匹配需要正确初始化
4. **dp[i][j-2] vs dp[i-1][j]**：分别代表匹配零个和匹配多个

### 6.2 两种星号的区别

```
正则表达式匹配：
  'a*' = 匹配零个或多个 'a'
  '.*' = 匹配任意字符串

通配符匹配：
  '*' = 匹配任意字符串
  '?' = 匹配任意单个字符
```

### 6.3 调试技巧

- 画出 dp 表格，逐步填写
- 先处理无 `*` 的简单情况
- 特别注意 `dp[0][j]` 的初始化
