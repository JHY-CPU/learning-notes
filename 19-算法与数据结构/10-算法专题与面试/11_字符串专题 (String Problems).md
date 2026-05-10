# 字符串专题 (String Problems)

## 一、核心概念

### 1.1 字符串基础

字符串是由字符组成的序列，在大多数语言中是不可变的。

**常见操作时间复杂度：**
- 访问：$O(1)$
- 查找/包含：$O(n)$
- 拼接：$O(n+m)$（不可变时）
- 子串：$O(m)$

### 1.2 字符串编码

- **ASCII：** 128个字符，7位编码
- **Unicode：** 通用字符集，UTF-8可变长度编码
- **面试中通常假设小写英文字母（26个字符）**

---

## 二、核心技巧

### 2.1 滑动窗口

```python
# 最长无重复字符子串 (LeetCode 3)
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

### 2.2 KMP算法

用于字符串匹配，时间复杂度 $O(n+m)$。

```python
def kmp_search(text, pattern):
    n, m = len(text), len(pattern)
    if m == 0: return 0

    # 构建next数组（失败函数）
    next_arr = [0] * m
    j = 0
    for i in range(1, m):
        while j > 0 and pattern[i] != pattern[j]:
            j = next_arr[j-1]
        if pattern[i] == pattern[j]:
            j += 1
        next_arr[i] = j

    # 匹配
    j = 0
    for i in range(n):
        while j > 0 and text[i] != pattern[j]:
            j = next_arr[j-1]
        if text[i] == pattern[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1
```

### 2.3 字符串哈希 (Rabin-Karp)

```python
def rabin_karp(text, pattern):
    n, m = len(text), len(pattern)
    base, mod = 26, 10**9 + 7

    # 计算pattern的哈希
    p_hash = 0
    for c in pattern:
        p_hash = (p_hash * base + ord(c)) % mod

    # 滑动窗口哈希
    t_hash = 0
    for i in range(m):
        t_hash = (t_hash * base + ord(text[i])) % mod

    if t_hash == p_hash and text[:m] == pattern:
        return 0

    power = pow(base, m-1, mod)
    for i in range(m, n):
        t_hash = ((t_hash - ord(text[i-m]) * power) * base + ord(text[i])) % mod
        if t_hash == p_hash and text[i-m+1:i+1] == pattern:
            return i - m + 1
    return -1
```

---

## 三、经典题目详解

### 3.1 最长回文子串 (LeetCode 5)

```python
# 中心扩展法
def longest_palindrome(s):
    def expand(l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1; r += 1
        return s[l+1:r]

    result = ""
    for i in range(len(s)):
        odd = expand(i, i)
        even = expand(i, i+1)
        result = max(result, odd, even, key=len)
    return result
```

### 3.2 最小覆盖子串 (LeetCode 76)

```python
from collections import Counter

def min_window(s, t):
    need, missing = Counter(t), len(t)
    left = start = end = 0

    for right, c in enumerate(s, 1):
        missing -= (need[c] > 0)
        need[c] -= 1

        if missing == 0:
            while left < right and need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if end == 0 or right - left < end - start:
                start, end = left, right

    return s[start:end]
```

### 3.3 字符串转换整数 (LeetCode 8)

```python
def my_atoi(s):
    s = s.strip()
    if not s: return 0

    sign, i, result = 1, 0, 0
    if s[0] in '+-':
        sign = 1 if s[0] == '+' else -1
        i = 1

    while i < len(s) and s[i].isdigit():
        result = result * 10 + int(s[i])
        i += 1

    result *= sign
    return max(-2**31, min(2**31 - 1, result))
```

### 3.4 字母异位词分组 (LeetCode 49)

```python
def group_anagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

### 3.5 最长公共前缀 (LeetCode 14)

```python
def longest_common_prefix(strs):
    if not strs: return ""
    prefix = strs[0]
    for s in strs[1:]:
        while not s.startswith(prefix):
            prefix = prefix[:-1]
            if not prefix: return ""
    return prefix
```

### 3.6 正则表达式匹配 (LeetCode 10)

```python
def is_match(s, p):
    m, n = len(s), len(p)
    dp = [[False]*(n+1) for _ in range(m+1)]
    dp[0][0] = True

    for j in range(2, n+1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m+1):
        for j in range(1, n+1):
            if p[j-1] == '*':
                dp[i][j] = dp[i][j-2] or (dp[i-1][j] and p[j-2] in (s[i-1], '.'))
            elif p[j-1] in (s[i-1], '.'):
                dp[i][j] = dp[i-1][j-1]

    return dp[m][n]
```

---

## 四、Trie前缀树

```python
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False

    def insert(self, word):
        node = self
        for c in word:
            if c not in node.children:
                node.children[c] = Trie()
            node = node.children[c]
        node.is_end = True

    def search(self, word):
        node = self._find(word)
        return node is not None and node.is_end

    def starts_with(self, prefix):
        return self._find(prefix) is not None

    def _find(self, word):
        node = self
        for c in word:
            if c not in node.children: return None
            node = node.children[c]
        return node
```

---

## 五、复杂度分析

| 算法 | 时间复杂度 | 空间复杂度 |
|------|-----------|-----------|
| 滑动窗口 | $O(n)$ | $O(k)$ k为字符集 |
| KMP | $O(n+m)$ | $O(m)$ |
| Rabin-Karp | $O(n+m)$ 平均 | $O(1)$ |
| Trie插入/查找 | $O(m)$ | $O(字符总数)$ |
| 最小覆盖子串 | $O(n)$ | $O(k)$ |

---

## 六、面试高频题

1. **LeetCode 3：** 无重复字符的最长子串
2. **LeetCode 5：** 最长回文子串
3. **LeetCode 76：** 最小覆盖子串
4. **LeetCode 8：** 字符串转换整数
5. **LeetCode 10：** 正则表达式匹配
6. **LeetCode 49：** 字母异位词分组
7. **LeetCode 14：** 最长公共前缀
8. **LeetCode 208：** 实现Trie
9. **LeetCode 28：** 实现strStr()（KMP）
10. **LeetCode 438：** 找到字符串中所有字母异位词
