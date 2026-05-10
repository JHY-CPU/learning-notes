# 回文串DP (Palindrome DP)

## 一、概念定义与原理

### 1.1 什么是回文串

回文串是指正读和反读都一样的字符串，例如 "aba"、"abba"、"racecar"。

回文串问题在算法面试和竞赛中非常常见，核心思路是利用**动态规划**来避免重复计算子串是否为回文的判断。

### 1.2 核心DP定义

设 `dp[i][j]` 表示子串 `s[i..j]` 是否为回文串（布尔值）。

**状态转移方程：**

```
dp[i][j] = (s[i] == s[j]) && dp[i+1][j-1]
```

**边界条件：**
- `dp[i][i] = true`（单个字符是回文）
- `dp[i][i+1] = (s[i] == s[i+1])`（两个相邻字符）

### 1.3 填表顺序

由于 `dp[i][j]` 依赖于 `dp[i+1][j-1]`，需要按**子串长度从小到大**填表，或者按 `i` 从大到小、`j` 从小到大遍历。

---

## 二、经典问题与算法

### 2.1 最长回文子串 (LeetCode 5)

**问题：** 给定字符串 `s`，找到 `s` 中最长的回文子串。

**思路：** 枚举所有子串，用DP判断是否为回文，记录最长的。

**时间复杂度：** $O(n^2)$
**空间复杂度：** $O(n^2)$

**中心扩展法优化：** 枚举每个中心点（奇数长度和偶数长度），向两边扩展，时间 $O(n^2)$，空间 $O(1)$。

### 2.2 最长回文子序列 (LeetCode 516)

**问题：** 找到字符串中最长的回文子序列长度（不要求连续）。

**DP定义：** `dp[i][j]` = `s[i..j]` 中最长回文子序列的长度。

**状态转移：**
```
if s[i] == s[j]:
    dp[i][j] = dp[i+1][j-1] + 2
else:
    dp[i][j] = max(dp[i+1][j], dp[i][j-1])
```

### 2.3 回文子串计数 (LeetCode 647)

**问题：** 统计字符串中有多少个回文子串。

**思路：** 用同样的 `dp[i][j]` 判断回文，每发现一个就计数加一。

### 2.4 分割回文串 (LeetCode 131)

**问题：** 将字符串分割成若干子串，使每个子串都是回文，返回所有可能的分割方案。

**思路：** 预处理DP表判断回文 + 回溯搜索所有分割方案。

### 2.5 分割回文串II (LeetCode 127)

**问题：** 最少切割次数使每个子串都是回文。

**DP定义：** `dp[i]` = 使 `s[0..i]` 每段都是回文的最少切割数。

**状态转移：**
```
dp[i] = min(dp[j] + 1)  for j < i, if s[j+1..i] is palindrome
```

---

## 三、代码实现

### 3.1 最长回文子串 - C++

```cpp
#include <bits/stdc++.h>
using namespace std;

// 方法一：动态规划 O(n^2)
string longestPalindrome(string s) {
    int n = s.size();
    if (n < 2) return s;

    vector<vector<bool>> dp(n, vector<bool>(n, false));
    int start = 0, maxLen = 1;

    // 初始化：单个字符
    for (int i = 0; i < n; i++) dp[i][i] = true;

    // 按长度递增填表
    for (int len = 2; len <= n; len++) {
        for (int i = 0; i + len - 1 < n; i++) {
            int j = i + len - 1;
            if (s[i] == s[j]) {
                if (len == 2 || dp[i+1][j-1]) {
                    dp[i][j] = true;
                    if (len > maxLen) {
                        maxLen = len;
                        start = i;
                    }
                }
            }
        }
    }
    return s.substr(start, maxLen);
}

// 方法二：中心扩展法 O(n^2) 时间, O(1) 空间
string longestPalindrome_expand(string s) {
    int n = s.size(), start = 0, maxLen = 1;

    auto expand = [&](int left, int right) {
        while (left >= 0 && right < n && s[left] == s[right]) {
            left--; right++;
        }
        int len = right - left - 1;
        if (len > maxLen) {
            maxLen = len;
            start = left + 1;
        }
    };

    for (int i = 0; i < n; i++) {
        expand(i, i);     // 奇数长度
        expand(i, i + 1); // 偶数长度
    }
    return s.substr(start, maxLen);
}
```

### 3.2 最长回文子序列 - C++

```cpp
int longestPalindromeSubseq(string s) {
    int n = s.size();
    vector<vector<int>> dp(n, vector<int>(n, 0));

    for (int i = n - 1; i >= 0; i--) {
        dp[i][i] = 1;
        for (int j = i + 1; j < n; j++) {
            if (s[i] == s[j])
                dp[i][j] = dp[i+1][j-1] + 2;
            else
                dp[i][j] = max(dp[i+1][j], dp[i][j-1]);
        }
    }
    return dp[0][n-1];
}
```

### 3.3 回文子串计数 - C++

```cpp
int countSubstrings(string s) {
    int n = s.size(), count = 0;
    vector<vector<bool>> dp(n, vector<bool>(n, false));

    for (int i = n - 1; i >= 0; i--) {
        for (int j = i; j < n; j++) {
            if (s[i] == s[j] && (j - i <= 2 || dp[i+1][j-1])) {
                dp[i][j] = true;
                count++;
            }
        }
    }
    return count;
}
```

### 3.4 分割回文串II - C++

```cpp
int minCut(string s) {
    int n = s.size();
    // 预处理回文表
    vector<vector<bool>> isPalin(n, vector<bool>(n, false));
    for (int i = n - 1; i >= 0; i--)
        for (int j = i; j < n; j++)
            if (s[i] == s[j] && (j - i <= 2 || isPalin[i+1][j-1]))
                isPalin[i][j] = true;

    // DP求最少切割
    vector<int> dp(n, INT_MAX);
    for (int i = 0; i < n; i++) {
        if (isPalin[0][i]) { dp[i] = 0; continue; }
        for (int j = 0; j < i; j++)
            if (isPalin[j+1][i])
                dp[i] = min(dp[i], dp[j] + 1);
    }
    return dp[n-1];
}
```

### 3.5 Python 实现

```python
def longest_palindrome(s: str) -> str:
    n = len(s)
    if n < 2:
        return s

    dp = [[False] * n for _ in range(n)]
    start, max_len = 0, 1

    for i in range(n):
        dp[i][i] = True

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if s[i] == s[j]:
                if length == 2 or dp[i+1][j-1]:
                    dp[i][j] = True
                    if length > max_len:
                        max_len = length
                        start = i

    return s[start:start + max_len]


def longest_palindrome_subseq(s: str) -> int:
    n = len(s)
    dp = [[0] * n for _ in range(n)]

    for i in range(n - 1, -1, -1):
        dp[i][i] = 1
        for j in range(i + 1, n):
            if s[i] == s[j]:
                dp[i][j] = dp[i+1][j-1] + 2
            else:
                dp[i][j] = max(dp[i+1][j], dp[i][j-1])

    return dp[0][n-1]


def count_palindromic_substrings(s: str) -> int:
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    count = 0

    for i in range(n - 1, -1, -1):
        for j in range(i, n):
            if s[i] == s[j] and (j - i <= 2 or dp[i+1][j-1]):
                dp[i][j] = True
                count += 1

    return count


# 测试
print(longestPalindrome("babad"))         # "bab" 或 "aba"
print(longest_palindrome_subseq("bbbab")) # 4 ("bbbb")
print(count_palindromic_substrings("aaa")) # 6
```

---

## 四、Manacher算法（进阶）

### 4.1 算法思想

Manacher算法能在 $O(n)$ 时间内找到最长回文子串。核心技巧：

1. **统一插入分隔符：** 将 "aba" 变为 "#a#b#a#"，统一奇偶长度处理
2. **利用对称性：** 维护已知最右回文边界，利用对称性跳过已计算的区域

### 4.2 代码实现

```cpp
string manacher(string s) {
    // 构造统一字符串
    string t = "#";
    for (char c : s) { t += c; t += "#"; }
    int n = t.size();

    vector<int> p(n, 0); // p[i]: 以i为中心的回文半径
    int center = 0, right = 0;

    for (int i = 0; i < n; i++) {
        if (i < right)
            p[i] = min(right - i, p[2 * center - i]);

        // 中心扩展
        while (i - p[i] - 1 >= 0 && i + p[i] + 1 < n
               && t[i - p[i] - 1] == t[i + p[i] + 1])
            p[i]++;

        // 更新边界
        if (i + p[i] > right) {
            center = i;
            right = i + p[i];
        }
    }

    // 找最大回文
    int maxLen = 0, idx = 0;
    for (int i = 0; i < n; i++)
        if (p[i] > maxLen) { maxLen = p[i]; idx = i; }

    int start = (idx - maxLen) / 2;
    return s.substr(start, maxLen);
}
```

---

## 五、复杂度分析

| 问题 | DP方法时间 | DP方法空间 | 优化方法 |
|------|-----------|-----------|---------|
| 最长回文子串 | $O(n^2)$ | $O(n^2)$ | Manacher $O(n)$ |
| 最长回文子序列 | $O(n^2)$ | $O(n^2)$ | 可优化到 $O(n)$ 空间 |
| 回文子串计数 | $O(n^2)$ | $O(n^2)$ | 中心扩展 $O(1)$ 空间 |
| 分割回文串II | $O(n^2)$ | $O(n^2)$ | — |
| 判断回文 | $O(n)$ | $O(1)$ | 双指针 |

---

## 六、竞赛与面试应用场景

1. **LeetCode 5：** 最长回文子串
2. **LeetCode 516：** 最长回文子序列
3. **LeetCode 647：** 回文子串
4. **LeetCode 131：** 分割回文串
5. **LeetCode 127：** 分割回文串II
6. **LeetCode 132：** 分割回文串III
7. **LeetCode 214：** 最短回文串（KMP + 回文判断）

**面试技巧：**
- 先写 $O(n^2)$ DP解法，确保正确
- 追问优化时提 Manacher 算法
- 回文子序列 vs 回文子串要分清（连续 vs 不连续）
