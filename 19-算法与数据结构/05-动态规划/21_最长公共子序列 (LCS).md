# 最长公共子序列 (LCS)

## 1. 概念与定义

最长公共子序列（Longest Common Subsequence，LCS）是指两个序列中都出现的最长子序列。子序列不要求连续，但要求保持相对顺序。

LCS的核心应用：
- 文件差异比较（diff工具）
- DNA序列比对
- 版本控制系统
- 编辑距离的基础

## 2. 状态定义与转移方程

### 2.1 标准DP

```
dp[i][j] = s1[0..i-1] 和 s2[0..j-1] 的最长公共子序列长度
转移：
  if s1[i-1] == s2[j-1]:
    dp[i][j] = dp[i-1][j-1] + 1
  else:
    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
初始条件：dp[i][0] = 0, dp[0][j] = 0
```

### 2.2 空间优化

```
dp[j] = 当前行的LCS长度
需要保存左上角的值（prev）
```

### 2.3 LCS与编辑距离的关系

```
编辑距离（只有插入和删除）= len(s1) + len(s2) - 2 * LCS
即：两个字符串删除到LCS需要的操作数
```

## 3. 算法实现

### 3.1 标准O(mn) DP

```python
def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
```

### 3.2 还原LCS路径

```python
def lcs_with_path(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯还原
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if text1[i - 1] == text2[j - 1]:
            lcs.append(text1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()
    return dp[m][n], ''.join(lcs)
```

### 3.3 空间优化O(min(m,n))

```python
def lcs_optimized(text1, text2):
    if len(text1) < len(text2):
        text1, text2 = text2, text1
    m, n = len(text1), len(text2)
    dp = [0] * (n + 1)

    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if text1[i - 1] == text2[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[n]
```

### 3.4 最长公共子串（连续）

```python
def longestCommonSubstring(s1, s2):
    """连续的公共子串"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                max_len = max(max_len, dp[i][j])
            else:
                dp[i][j] = 0  # 不连续则重置为0

    return max_len
```

### 3.5 C++ 实现

```cpp
int longestCommonSubsequence(string text1, string text2) {
    int m = text1.size(), n = text2.size();
    vector<vector<int>> dp(m+1, vector<int>(n+1, 0));
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            if (text1[i-1] == text2[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]);
    return dp[m][n];
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| LCS（标准） | O(mn) | O(mn) |
| LCS（空间优化） | O(mn) | O(min(m,n)) |
| 最长公共子串 | O(mn) | O(mn) |
| 还原路径 | O(mn) | O(mn) |

## 5. 典型例题

### 例题1：不相交的线（LeetCode 1035）

```python
def maxUncrossedLines(nums1, nums2):
    """本质就是求LCS"""
    return lcs_optimized(nums1, nums2)
```

### 例题2：两个字符串的最小ASCII删除和（LeetCode 712）

```python
def minimumDeleteSum(s1, s2):
    """删除字符使两串相等，使删除的ASCII和最小"""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + ord(s1[i - 1])
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + ord(s2[j - 1])

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + ord(s1[i - 1]),
                    dp[i][j - 1] + ord(s2[j - 1])
                )

    return dp[m][n]
```

### 例题3：最长重复子数组（LeetCode 718）

```python
def findLength(nums1, nums2):
    """最长公共子数组（连续）"""
    m, n = len(nums1), len(nums2)
    dp = [0] * (n + 1)
    max_len = 0

    for i in range(1, m + 1):
        # 逆序遍历，避免覆盖
        for j in range(n, 0, -1):
            if nums1[i - 1] == nums2[j - 1]:
                dp[j] = dp[j - 1] + 1
                max_len = max(max_len, dp[j])
            else:
                dp[j] = 0

    return max_len
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **子序列 vs 子串**：子序列不连续，子串连续
2. **索引偏移**：dp[i][j] 对应 s1[i-1] 和 s2[j-1]
3. **空间优化时方向**：子串问题空间优化需要逆序

### 6.2 扩展问题

- **多个序列的LCS**：多维DP，时间复杂度 O(2^k * n^k)
- **LCS长度为k的方案数**：计数DP
- **编辑距离**：LCS是编辑距离的特例（只有删除操作）

### 6.3 与LIS的关系

如果 s2 是 s1 排序后的版本，LCS 等价于 LIS。更一般地，LIS 可以转化为 LCS：
```
LIS(arr) = LCS(arr, sorted(arr))
```
但这个转化的时间复杂度是 O(n²)，不如直接用 O(nlogn) 的LIS算法。
