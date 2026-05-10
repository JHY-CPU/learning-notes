# 单词拆分 (Word Break)

## 1. 概念与定义

单词拆分问题：给定一个字符串 s 和一个字典 wordDict，判断 s 是否可以被拆分为一个或多个字典中单词的序列。

这是一个经典的完全背包DP问题 — 字典中的单词可以重复使用（在不同位置）。

相关题目：
- LeetCode 139：判断能否拆分
- LeetCode 140：返回所有拆分方案
- LeetCode 140是139的进阶版，需要回溯

## 2. 状态定义与转移方程

```
dp[i] = s[0:i] 是否可以被拆分
dp[i] = any(dp[j] and s[j:i] in wordDict) for j in [0, i)
dp[0] = True（空串可以被拆分）
```

### 优化思路

- 按单词长度枚举，而不是枚举所有 j
- 使用Trie树优化前缀匹配
- BFS + 剪枝

## 3. 算法实现

### 3.1 基本DP

```python
def wordBreak(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break

    return dp[n]
```

### 3.2 按单词长度优化

```python
def wordBreak_optimized(s, wordDict):
    word_set = set(wordDict)
    max_len = max(len(w) for w in wordDict) if wordDict else 0
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True

    for i in range(1, n + 1):
        for length in range(1, min(i, max_len) + 1):
            if dp[i - length] and s[i - length:i] in word_set:
                dp[i] = True
                break

    return dp[n]
```

### 3.3 返回所有方案（LeetCode 140）

```python
def wordBreakII(s, wordDict):
    word_set = set(wordDict)
    n = len(s)
    dp = [[] for _ in range(n + 1)]
    dp[0] = [[]]

    for i in range(1, n + 1):
        for word in word_set:
            if i >= len(word) and dp[i - len(word)]:
                if s[i - len(word):i] == word:
                    for prev in dp[i - len(word)]:
                        dp[i].append(prev + [word])

    return [' '.join(words) for words in dp[n]]
```

### 3.4 记忆化搜索 + 回溯

```python
def wordBreakII_memo(s, wordDict):
    word_set = set(wordDict)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def dfs(s):
        if not s:
            return [[]]
        result = []
        for word in word_set:
            if s.startswith(word):
                for rest in dfs(s[len(word):]):
                    result.append([word] + rest)
        return result

    return [' '.join(words) for words in dfs(s)]
```

### 3.5 单词拆分II（带剪枝）

```python
def wordBreakII_pruned(s, wordDict):
    word_set = set(wordDict)
    max_len = max(len(w) for w in wordDict)

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def can_break(s):
        """预判能否拆分，用于剪枝"""
        if not s:
            return True
        for length in range(1, min(len(s), max_len) + 1):
            if s[:length] in word_set and can_break(s[length:]):
                return True
        return False

    result = []
    def backtrack(start, path):
        if start == len(s):
            result.append(' '.join(path))
            return
        for end in range(start + 1, min(len(s) + 1, start + max_len + 1)):
            word = s[start:end]
            if word in word_set and can_break(s[end:]):
                backtrack(end, path + [word])

    backtrack(0, [])
    return result
```

## 4. 复杂度分析

| 问题 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 单词拆分（139） | O(n * maxLen * k) | O(n) |
| 单词拆分II（140） | O(n * 方案数) | O(n * 方案数) |
| 记忆化搜索 | O(n²) + 方案数 | O(n) |

其中 k 为字符串比较的时间。

## 5. 典型例题

### 例题1：单词拆分II（LeetCode 140）

```python
# 见 wordBreakII_pruned 实现
```

### 例题2：连接词（LeetCode 472）

```python
def findAllConcatenatedWordsInADict(words):
    """找所有能由其他单词拼接而成的单词"""
    word_set = set(words)
    result = []

    def can_form(word):
        if not word:
            return False
        n = len(word)
        dp = [False] * (n + 1)
        dp[0] = True
        for i in range(1, n + 1):
            start = 0 if i == n else 1  # 至少两个单词
            for j in range(start, i):
                if dp[j] and word[j:i] in word_set:
                    dp[i] = True
                    break
        return dp[n]

    for word in sorted(words, key=len):
        if can_form(word):
            result.append(word)

    return result
```

## 6. 常见陷阱与优化

### 6.1 常见陷阱

1. **超时**：140题如果没有预判剪枝会超时
2. **重复单词**：字典中可能有重复单词，用set去重
3. **空串**：空串能否被拆分？通常可以（返回True/[""]）
4. **方案数爆炸**：如果拆分方案很多，返回结果可能很大

### 6.2 优化策略

1. **长度限制**：只枚举可能的单词长度
2. **预判剪枝**：先判断后半部分能否拆分
3. **Trie树**：用Trie树加速前缀匹配
4. **BFS**：对于139题，BFS可能比DP更直观
