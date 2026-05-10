# 编辑距离 (Edit Distance)

## 1. 概念与定义

编辑距离（Edit Distance），也称 Levenshtein 距离，是将一个字符串转换为另一个字符串所需的最少操作次数。允许的操作有三种：
1. **插入**一个字符
2. **删除**一个字符
3. **替换**一个字符

编辑距离是衡量两个字符串相似度的重要指标，广泛应用于：
- 拼写检查和纠错
- DNA序列比对
- 自然语言处理
- diff工具

## 2. 状态定义与转移方程

### 2.1 标准DP

```
dp[i][j] = 将 word1[0..i-1] 转换为 word2[0..j-1] 的最少操作数

转移：
  if word1[i-1] == word2[j-1]:
    dp[i][j] = dp[i-1][j-1]  # 字符相同，不需要操作
  else:
    dp[i][j] = 1 + min(
      dp[i-1][j],    # 删除 word1[i-1]
      dp[i][j-1],    # 插入 word2[j-1]
      dp[i-1][j-1]   # 替换 word1[i-1] 为 word2[j-1]
    )

初始条件：
  dp[i][0] = i  # 删除i个字符
  dp[0][j] = j  # 插入j个字符
```

### 2.2 理解三种操作

```
删除：dp[i-1][j] + 1 — word1少一个字符，word2不变
插入：dp[i][j-1] + 1 — word1不变，word2少一个字符
替换：dp[i-1][j-1] + 1 — 两个都少一个字符
```

### 2.3 空间优化

```
dp[j] = 当前行的编辑距离
需要保存左上角值（dp[j-1]在更新前的值）
```

## 3. 算法实现

### 3.1 标准O(mn) DP

```python
def minDistance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # 删除
                    dp[i][j - 1],    # 插入
                    dp[i - 1][j - 1] # 替换
                )

    return dp[m][n]
```

### 3.2 空间优化O(min(m,n))

```python
def minDistance_optimized(word1, word2):
    if len(word1) < len(word2):
        word1, word2 = word2, word1

    m, n = len(word1), len(word2)
    dp = list(range(n + 1))

    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if word1[i - 1] == word2[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(dp[j], dp[j - 1], prev)
            prev = temp

    return dp[n]
```

### 3.3 还原操作序列

```python
def minDistance_with_ops(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    # 回溯还原操作
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and word1[i-1] == word2[j-1]:
            i -= 1; j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            ops.append(f"Replace '{word1[i-1]}' at {i-1} with '{word2[j-1]}'")
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            ops.append(f"Delete '{word1[i-1]}' at {i-1}")
            i -= 1
        else:
            ops.append(f"Insert '{word2[j-1]}' at {i}")
            j -= 1

    ops.reverse()
    return dp[m][n], ops
```

### 3.4 只允许插入和删除

```python
def minInsertDelete(word1, word2):
    """只允许插入和删除的编辑距离"""
    # 等价于 m + n - 2 * LCS
    lcs = longestCommonSubsequence(word1, word2)
    return len(word1) + len(word2) - 2 * lcs

def longestCommonSubsequence(s1, s2):
    m, n = len(s1), len(s2)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i-1] == s2[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = temp
    return dp[n]
```

### 3.5 C++ 实现

```cpp
int minDistance(string word1, string word2) {
    int m = word1.size(), n = word2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (word1[i-1] == word2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
    return dp[m][n];
}
```

## 4. 复杂度分析

| 方法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 标准DP | O(mn) | O(mn) |
| 空间优化 | O(mn) | O(min(m,n)) |
| 还原操作 | O(mn) | O(mn) |

## 5. 典型例题

### 例题1：一次编辑（LeetCode 16.05）

```python
def oneEditAway(first, second):
    """判断是否可以通过一次编辑将first变为second"""
    m, n = len(first), len(second)
    if abs(m - n) > 1:
        return False

    if m > n:
        first, second = second, first
        m, n = n, m

    found_diff = False
    i = j = 0
    while i < m and j < n:
        if first[i] != second[j]:
            if found_diff:
                return False
            found_diff = True
            if m == n:
                i += 1  # 替换
        else:
            i += 1
        j += 1
    return True
```

### 例题2：交错字符串（LeetCode 97）

```python
def isInterleave(s1, s2, s3):
    """判断s3是否由s1和s2交错组成"""
    m, n, l = len(s1), len(s2), len(s3)
    if m + n != l:
        return False

    dp = [[False] * (n + 1) for _ in range(m + 1)]
    dp[0][0] = True

    for i in range(m + 1):
        for j in range(n + 1):
            if i > 0 and s1[i-1] == s3[i+j-1]:
                dp[i][j] |= dp[i-1][j]
            if j > 0 and s2[j-1] == s3[i+j-1]:
                dp[i][j] |= dp[i][j-1]

    return dp[m][n]
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **初始化遗漏**：dp[i][0] = i, dp[0][j] = j
2. **三种操作都要考虑**：取 min 时不要遗漏
3. **空间优化时保存prev**：一维数组需要保存左上角

### 6.2 扩展变体

- **不同操作不同代价**：插入、删除、替换代价不同
- **只允许插入**：编辑距离 = n - LCS
- **只允许替换**：Hamming距离（两串等长时）
- **大写转换不计操作**：预处理将两串统一大小写

### 6.3 与其他问题的关系

```
编辑距离（Levenshtein）= 插入 + 删除 + 替换
只插入和删除 = m + n - 2*LCS
LCS = (m + n - 编辑距离(只有插入删除)) / 2
Hamming距离 = 不等位置数（要求等长）
```
