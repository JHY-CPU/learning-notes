# 模拟面试-字符串 (Mock Interview - String)

## 一、面试流程模拟

**时间：** 45分钟
**题目数：** 2-3题

---

## 二、题目1：无重复字符的最长子串 (LeetCode 3, Medium, 15分钟)

### 题目描述

给定字符串 `s`，找出不含有重复字符的最长子串的长度。

### 面试过程

**候选人：**
"滑动窗口方法。维护一个窗口 `[left, right]`，窗口内没有重复字符。用哈希表记录每个字符最后出现的位置。

当遇到重复字符时，将 `left` 跳到重复字符的下一个位置。"

### 代码

```python
def length_of_longest_substring(s):
    seen = {}
    left = max_len = 0

    for right, c in enumerate(s):
        if c in seen and seen[c] >= left:
            left = seen[c] + 1
        seen[c] = right
        max_len = max(max_len, right - left + 1)

    return max_len
```

**面试官追问：** "如果字符集很大怎么办？"
"哈希表天然支持大字符集，不受影响。如果字符集很小（如只有小写字母），可以用数组代替哈希表，常数更优。"

**复杂度：** 时间 $O(n)$，空间 $O(\min(n, |\Sigma|))$。

---

## 三、题目2：最长回文子串 (LeetCode 5, Medium, 15分钟)

### 面试过程

**候选人：**
"三种方法：
1. **DP：** `dp[i][j]` 表示 `s[i..j]` 是否为回文，O(n^2) 时间空间
2. **中心扩展：** 枚举每个中心点，向两边扩展，O(n^2) 时间 O(1) 空间
3. **Manacher：** O(n) 但较复杂

我用中心扩展法。"

### 代码

```python
def longest_palindrome(s):
    def expand(left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1; right += 1
        return s[left+1:right]

    result = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i + 1)
        result = max(result, odd, even, key=len)
    return result
```

---

## 四、题目3：编辑距离 (LeetCode 72, Hard, 15分钟)

### 面试过程

**候选人：**
"经典DP问题。`dp[i][j]` 表示 `word1[:i]` 变成 `word2[:j]` 的最少操作数。

三种操作：
- 插入：`dp[i][j-1] + 1`
- 删除：`dp[i-1][j] + 1`
- 替换：`dp[i-1][j-1] + 1`（字符相同时不用替换）"

### 代码

```python
def min_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0]*(n+1) for _ in range(m+1)]

    for i in range(m+1): dp[i][0] = i
    for j in range(n+1): dp[0][j] = j

    for i in range(1, m+1):
        for j in range(1, n+1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1

    return dp[m][n]
```

**面试官追问：** "空间能优化吗？"
"可以优化到 O(min(m,n))，因为 dp[i][j] 只依赖上一行和当前行。"

---

## 五、评分要点

1. **滑动窗口是否熟练** — 字符串最核心的技巧
2. **边界处理意识** — 空串、单字符、全相同字符
3. **DP状态定义** — 能否快速定义正确的状态
4. **优化思路** — 空间优化、Manacher算法等
