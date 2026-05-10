# 回文DP (Palindrome DP)

## 1. 概念与定义

回文DP是与回文串相关的动态规划问题。回文串是正读和反读都相同的字符串。回文DP主要包括：
- **最长回文子串**：给定字符串，找最长的回文连续子串
- **最长回文子序列**：给定字符串，找最长的回文子序列（不要求连续）
- **回文分割**：将字符串分割成若干回文子串
- **回文计数**：统计字符串中回文子串的个数

回文问题的核心性质：**如果 s[i] == s[j]，那么 s[i..j] 是回文当且仅当 s[i+1..j-1] 是回文。**

## 2. 状态定义与转移方程

### 2.1 最长回文子串

```
dp[i][j] = s[i..j] 是否为回文串
转移：
  dp[i][j] = (s[i] == s[j]) and (j - i <= 2 or dp[i+1][j-1])
  - j - i <= 2：长度 <= 3 时，首尾相同就是回文
  - dp[i+1][j-1]：去掉首尾后是否为回文

枚举顺序：i从大到小，j从小到大（或按长度枚举）
答案：max(j - i + 1) for dp[i][j] == True
```

### 2.2 最长回文子序列

```
dp[i][j] = s[i..j] 的最长回文子序列长度
转移：
  if s[i] == s[j]:
    dp[i][j] = dp[i+1][j-1] + 2
  else:
    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
初始条件：dp[i][i] = 1
答案：dp[0][n-1]
```

### 2.3 回文分割

```
dp[i] = s[0..i] 的最少分割次数
转移：
  dp[i] = min(dp[j] + 1) for j in [0, i] where s[j+1..i] is palindrome
  需要预处理 is_palindrome[i][j]
```

### 2.4 回文子串计数

```
方法1：dp[i][j] = s[i..j] 是否为回文，统计所有 dp[i][j] == True
方法2：中心扩展法，对每个中心向两边扩展，O(n²)
```

## 3. 算法实现

### 3.1 最长回文子串（LeetCode 5）

```python
def longestPalindrome(s):
    """DP方法求最长回文子串"""
    n = len(s)
    if n < 2:
        return s

    dp = [[False] * n for _ in range(n)]
    max_len = 1
    start = 0

    # 单个字符都是回文
    for i in range(n):
        dp[i][i] = True

    # 按长度递推
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] != s[j]:
                dp[i][j] = False
            else:
                if length <= 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]

            if dp[i][j] and length > max_len:
                max_len = length
                start = i

    return s[start:start + max_len]

# 中心扩展法 O(n²) 时间 O(1) 空间
def longestPalindrome_expand(s):
    n = len(s)
    if n < 2:
        return s

    def expand(left, right):
        while left >= 0 and right < n and s[left] == s[right]:
            left -= 1
            right += 1
        return left + 1, right - 1

    start, max_len = 0, 1
    for i in range(n):
        l1, r1 = expand(i, i)       # 奇数长度
        l2, r2 = expand(i, i + 1)   # 偶数长度
        if r1 - l1 + 1 > max_len:
            max_len = r1 - l1 + 1
            start = l1
        if r2 - l2 + 1 > max_len:
            max_len = r2 - l2 + 1
            start = l2

    return s[start:start + max_len]
```

### 3.2 最长回文子序列（LeetCode 516）

```python
def longestPalindromeSubseq(s):
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = 1

    # 注意遍历方向：i从大到小，j从小到大
    for i in range(n - 1, -1, -1):
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i + 1][j - 1] + 2
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j - 1])

    return dp[0][n - 1]

# 空间优化：转化为LCS问题
def longestPalindromeSubseq_lcs(s):
    """最长回文子序列 = s 和 reverse(s) 的最长公共子序列"""
    def lcs(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]
    return lcs(s, s[::-1])
```

### 3.3 回文分割II（LeetCode 132）

```python
def minCut(s):
    """
    将s分割成回文子串的最少切割次数
    """
    n = len(s)

    # 预处理：判断 s[i..j] 是否为回文
    is_palindrome = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palindrome[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length <= 3:
                    is_palindrome[i][j] = True
                else:
                    is_palindrome[i][j] = is_palindrome[i + 1][j - 1]

    # DP求最少分割
    dp = [float('inf')] * n
    for i in range(n):
        if is_palindrome[0][i]:
            dp[i] = 0
        else:
            for j in range(1, i + 1):
                if is_palindrome[j][i]:
                    dp[i] = min(dp[i], dp[j - 1] + 1)

    return dp[n - 1]
```

### 3.4 回文子串计数（LeetCode 647）

```python
def countSubstrings(s):
    n = len(s)
    count = 0
    dp = [[False] * n for _ in range(n)]

    for i in range(n):
        dp[i][i] = True
        count += 1

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length <= 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i + 1][j - 1]
            if dp[i][j]:
                count += 1

    return count

# 中心扩展法更简洁
def countSubstrings_expand(s):
    count = 0
    n = len(s)
    for center in range(2 * n - 1):
        left = center // 2
        right = left + center % 2
        while left >= 0 and right < n and s[left] == s[right]:
            count += 1
            left -= 1
            right += 1
    return count
```

### 3.5 C++ 实现

```cpp
// 最长回文子串
string longestPalindrome(string s) {
    int n = s.size();
    vector<vector<bool>> dp(n, vector<bool>(n, false));
    int start = 0, maxLen = 1;
    for (int i = 0; i < n; i++) dp[i][i] = true;
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i <= n - len; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                dp[i][j] = (len <= 3) ? true : dp[i+1][j-1];
            }
            if (dp[i][j] && len > maxLen) {
                maxLen = len;
                start = i;
            }
        }
    }
    return s.substr(start, maxLen);
}
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 | 优化方法 |
|------|-----------|-----------|---------|
| 最长回文子串（DP） | O(n²) | O(n²) | 中心扩展O(1)空间 |
| 最长回文子串（Manacher） | O(n) | O(n) | Manacher算法 |
| 最长回文子序列 | O(n²) | O(n²) | 转化为LCS |
| 回文分割 | O(n²) | O(n²) | 预处理回文表 |
| 回文计数 | O(n²) | O(n²) | 中心扩展O(1)空间 |

## 5. 典型例题

### 例题1：分割回文串（LeetCode 131）

```python
def partition(s):
    """返回所有可能的回文分割方案"""
    n = len(s)
    is_palindrome = [[False] * n for _ in range(n)]
    for i in range(n):
        is_palindrome[i][i] = True
    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                is_palindrome[i][j] = length <= 3 or is_palindrome[i + 1][j - 1]

    result = []
    def backtrack(start, path):
        if start == n:
            result.append(path[:])
            return
        for end in range(start, n):
            if is_palindrome[start][end]:
                path.append(s[start:end + 1])
                backtrack(end + 1, path)
                path.pop()

    backtrack(0, [])
    return result
```

### 例题2：回文对（LeetCode 336）

```python
def palindromePairs(words):
    """找所有拼接后为回文的单词对"""
    word_map = {w: i for i, w in enumerate(words)}
    result = []

    for i, word in enumerate(words):
        for j in range(len(word) + 1):
            prefix = word[:j]
            suffix = word[j:]

            # 情况1：suffix是回文，找reverse(prefix)
            if suffix == suffix[::-1]:
                target = prefix[::-1]
                if target in word_map and word_map[target] != i:
                    result.append([i, word_map[target]])

            # 情况2：prefix是回文，找reverse(suffix)
            if j > 0 and prefix == prefix[::-1]:
                target = suffix[::-1]
                if target in word_map and word_map[target] != i:
                    result.append([word_map[target], i])

    return list(set(tuple(r) for r in result))
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **遍历方向**：回文DP需要保证 `dp[i+1][j-1]` 已经计算，所以 i 要从大到小
2. **长度为2的特殊情况**：`s[i] == s[j]` 且 `j == i+1` 时直接是回文
3. **子串vs子序列**：
   - 子串：必须连续
   - 子序列：可以不连续
4. **空串处理**：空串是回文

### 6.2 优化方法

1. **Manacher算法**：O(n) 时间求最长回文子串
2. **中心扩展**：O(n²) 时间 O(1) 空间
3. **预处理回文表**：多次查询时先打表
4. **转化为LCS**：最长回文子序列 = s 和 reverse(s) 的 LCS

### 6.3 Manacher算法简介

```python
def manacher(s):
    """O(n) 求最长回文子串"""
    # 预处理：在字符间插入 #
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n  # p[i] = 以i为中心的最长回文半径
    center = right = 0

    for i in range(n):
        if i < right:
            p[i] = min(right - i, p[2 * center - i])
        # 扩展
        while i - p[i] - 1 >= 0 and i + p[i] + 1 < n and t[i - p[i] - 1] == t[i + p[i] + 1]:
            p[i] += 1
        # 更新中心和右边界
        if i + p[i] > right:
            center, right = i, i + p[i]

    max_len = max(p)
    center_idx = p.index(max_len)
    start = (center_idx - max_len) // 2
    return s[start:start + max_len]
```
